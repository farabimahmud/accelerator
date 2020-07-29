import sys
import math
import numpy as np
from copy import deepcopy
import logging

sys.path.append('booksim2/src')

from sim_object import SimObject
import pybooksim
from npu import NPU
from eventq import EventQueue
from message_buffer import *

logger = logging.getLogger(__name__)

class HMC(SimObject):
    # like static variables, reducing simulation time for data-parallel training
    inference_cycles = None
    training_cycles = None
    model_aggregation_cycles = None
    allreduce_aggregation_cycles = {}
    cur_mid = 0 # global message ID
    # XXX: oracle lockstep
    allreduce_timestep = None
    allreduce_remaining_for_timestep = None
    hmcs = []

    def __init__(self, i, args, eventq):
        super().__init__(eventq)
        HMC.hmcs.append(self)

        self.id = i
        self.name = 'HMC-{}'.format(self.id)
        self.args = args
        self.npu = NPU(args)
        self.num_npus = self.args.num_vaults

        self.local_eventq = EventQueue()
        self.computation_state = None
        self.communication_state = None

        self.compute_cycles = 0
        self.allreduce_compute_cycles = 0

        self.from_network_message_buffers = None
        self.to_network_message_buffers = None

        self.model = None
        self.bytes_per_param = 4 # bytes
        self.samples_per_npu = math.ceil(self.args.mini_batch_size / (self.args.num_vaults * self.args.num_hmcs))

        self.message_size = args.message_size # bytes
        self.sub_message_size = args.sub_message_size
        self.base_num_messages = None
        self.base_num_sub_messages = None
        self.num_messages = [None for i in range(self.args.radix)]
        self.num_sub_messages = [None for i in range(self.args.radix)]

        # for the schedule semantics, refer to allreduce/allreduce.py
        self.allreduce = None
        self.reduce_scatter_schedule = None
        self.all_gather_schedule = None
        self.new_step = True
        self.estimated_steptime = None
        self.estimated_next_steptime = None

        self.cur_mids = np.zeros(self.args.radix, dtype=int)
        for i in range(self.args.radix):
            self.cur_mids[i] = HMC.cur_mid
            HMC.cur_mid = (HMC.cur_mid + 1) % 2147483647
        self.sending = [None for i in range(self.args.radix)]
        self.free_nis = set([i for i in range(self.args.radix)])
        self.just_allocated_nis = {}
        self.pending_aggregations = []
        # the local accelerator can only control what to send but not what to receive
        self.messages_sent = [0] * self.args.radix
        self.messages_received = {'reduce-scatter': [{} for i in range(self.args.num_hmcs)],
                                  'all-gather': [0] * self.args.num_hmcs}
        self.messages_received_end = {'reduce-scatter': [{} for i in range(self.args.num_hmcs)],
                                      'all-gather': [False] * self.args.num_hmcs}
        self.total_messages_sent = 0


    '''
    load_model() - assign the NN model to this hmc
    @model: the NN model to be loaded
    '''
    def load_model(self, model):
        self.model = model
        if self.message_size == 0:
            self.base_num_messages = 1
            self.base_num_sub_messages = math.ceil(self.model.size * self.bytes_per_param /
                    self.sub_message_size / self.args.num_hmcs)
            # 58 is from message-buffer-size of 32, booksim-message-buffer size 16,
            # and booksim-injection-queue of 80-flit depth
            if self.base_num_sub_messages <= 58:
                self.estimated_steptime = self.base_num_sub_messages
            else:
                self.estimated_steptime = self.base_num_sub_messages * 16 - 58 * 16
        else:
            assert self.message_size >= self.sub_message_size
            self.base_num_messages = math.ceil(self.model.size * self.bytes_per_param /
                    self.message_size / self.args.num_hmcs)
            self.base_num_sub_messages = math.ceil(self.message_size / self.sub_message_size)
            if self.base_num_messages <= 58:
                self.estimated_steptime = self.base_num_messages
            else:
                self.estimated_steptime = self.base_num_messages * 17 - 58 * 16
    # end of load_model()


    '''
    startup() - startup function for simulation of HMC

    desc - schedule the start event for the simulation of HMC. Currently, assuming
           we are doing training only.
    TODO: should be extended later for more functionalities
    '''
    def startup(self):
        # currently start from training
        self.communication_state = 'idle'
        if self.args.only_compute:
            self.computation_state = 'idle'
            self.local_eventq.schedule('training', 0)
        elif self.args.only_allreduce or self.args.only_reduce_scatter:
            self.computation_state = 'aggregating'
            self.local_eventq.schedule('finish-aggregation', 0)
        elif self.args.only_all_gather:
            self.computation_state = 'idle'
            self.communication_state = 'all-gather'
            self.local_eventq.schedule('all-gather', 0)
        self.global_eventq.schedule(self, 0)
    # end of startup()


    '''
    set_allreduce() - set allreduce schedule
    @allreduce: allreduce schedule
    '''
    def set_allreduce(self, allreduce):
        if self.args.oracle_lockstep:
            if HMC.allreduce_timestep == None:
                HMC.allreduce_timestep = 0
                HMC.allreduce_remaining_for_timestep = [0] * allreduce.timesteps
        self.allreduce = allreduce
        self.reduce_scatter_schedule = deepcopy(allreduce.reduce_scatter_schedule[self.id])
        self.all_gather_schedule = deepcopy(allreduce.all_gather_schedule[self.id])
        if self.args.oracle_lockstep:
            for i, schedules in enumerate(self.reduce_scatter_schedule):
                if schedules == None:
                    continue
                for fl, schedule in schedules.items():
                    if schedule[2] != 0:
                        HMC.allreduce_remaining_for_timestep[schedule[3]] += 1
        for root in range(self.args.num_hmcs):
            for child in allreduce.trees_children[root][self.id]:
                self.messages_received['reduce-scatter'][root][child] = 0
                self.messages_received_end['reduce-scatter'][root][child] = False
    # end of set_allreduce()


    '''
    set_message_buffers() - set message buffers connected with network
    @from_network_message_buffers: message buffers for incoming messages
    @to_network_message_buffers: message buffers for outgoing messages
    '''
    def set_message_buffers(self, from_network_message_buffers, to_network_message_buffers):
        assert len(from_network_message_buffers) == self.args.radix
        assert len(to_network_message_buffers) == self.args.radix
        self.from_network_message_buffers = from_network_message_buffers
        self.to_network_message_buffers = to_network_message_buffers
    # end of set_message_buffers


    '''
    schedule() - schedule the event at a given time
    @event: the event to be scheduled
    @cycle: scheduled time
    '''
    def schedule(self, event, cycle):
        self.local_eventq.schedule(event, cycle)
        self.global_eventq.schedule(self, cycle)
    # end of schedule()


    '''
    reschedule() - reschedule the event due to structure hazard
    @event: the event to be rescheduled
    '''
    def reschedule(self, event):
        next_cycle = self.local_eventq.next_event_cycle()
        self.local_eventq.schedule(event, next_cycle)
    # end of reschedule()


    '''
    process() - event processing function in a particular cycle
    @cur_cycle: the current cycle that with events to be processed
    '''
    def process(self, cur_cycle):
        events = self.local_eventq.get_events(cur_cycle)

        # Evaluate the events
        for event in events:
            if event == 'training':
                self.training_evaluate(cur_cycle)
            elif event == 'finish-training':
                self.finish_training_evaluate(cur_cycle)
            elif event == 'aggregation':
                self.aggregation_evaluate(cur_cycle)
            elif event == 'finish-aggregation':
                self.finish_aggregation_evaluate(cur_cycle)
            elif event == 'reduce-scatter':
                self.reduce_scatter_evaluate(cur_cycle)
            elif event == 'send-reduce-message':
                self.send_reduce_message_evaluate(cur_cycle)
            elif event == 'all-gather':
                self.all_gather_evaluate(cur_cycle)
            elif event == 'send-gather-message':
                self.send_gather_message_evaluate(cur_cycle)
            elif event == 'incoming-message':
                self.incoming_message_evaluate(cur_cycle)
            else:
                raise RuntimeError('Unknown event type {} for {}'.format(event, self.name))

        # Update the states according to the events
        for event in events:
            if event == 'training':
                self.training_update(cur_cycle)
            elif event == 'finish-training':
                self.finish_training_update(cur_cycle)
            elif event == 'aggregation':
                self.aggregation_update(cur_cycle)
            elif event == 'finish-aggregation':
                self.finish_aggregation_update(cur_cycle)
            elif event == 'reduce-scatter':
                self.reduce_scatter_update(cur_cycle)
            elif event == 'send-reduce-message':
                self.send_reduce_message_update(cur_cycle)
            elif event == 'all-gather':
                self.all_gather_update(cur_cycle)
            elif event == 'send-gather-message':
                self.send_gather_message_update(cur_cycle)
            else:
                assert event == 'incoming-message'
                self.incoming_message_update(cur_cycle)
    # end of process()


    '''
    training_evaluate() - change to transient state
    '''
    def training_evaluate(self, cur_cycle):
        assert self.computation_state == 'idle'
        self.computation_state = 'idle-to-training'
    # end of training_evaluate()

    '''
    training_update() - update the state for training action
    '''
    def training_update(self, cur_cycle):
        assert self.computation_state == 'idle-to-training'
        self.computation_state = 'training'
        cycles = self.train()
        self.schedule('finish-training', cur_cycle + cycles)
        logger.info('{} | {} | starts training, computation state: {}, communication state: {}'.format(cur_cycle, self.name, self.computation_state, self.communication_state))
    # end of training_update()


    '''
    finish_training_evaluate() - change to transient state
    '''
    def finish_training_evaluate(self, cur_cycle):
        assert self.computation_state == 'training'
        self.computation_state = 'training-to-idle'
    # end of finish_training_evaluate()

    '''
    finish_training_update() - update the state and schedule dependent events
    '''
    def finish_training_update(self, cur_cycle):
        assert self.computation_state == 'training-to-idle'
        self.computation_state = 'idle'
        self.schedule('aggregation', cur_cycle + 1)
        logger.info('{} | {} finishes training, computation sate: {}, communication state: {}'.format(cur_cycle, self.name, self.computation_state, self.communication_state))
    # end of finish_training_update()


    '''
    aggregation_evaluate() - change to transient state
    '''
    def aggregation_evaluate(self, cur_cycle):
        if self.computation_state == 'idle':
            self.computation_state = 'idle-to-aggregating'
    # end of aggregation_evaluate()

    '''
    aggregation_update() - action execution
    '''
    def aggregation_update(self, cur_cycle):
        if self.computation_state == 'idle-to-aggregating':
            self.computation_state = 'aggregating'
            cycles = self.aggregate()
            self.schedule('finish-aggregation', cur_cycle + cycles)
            logger.info('{} | {} | starts aggregation, computation state: {}, communication state: {}'.format(cur_cycle, self.name, self.computation_state, self.communication_state))
        else:
            self.reschedule('aggregation')
            logger.debug('{} | {} | compute is not available for aggregation, state {}'.format(cur_cycle, self.name, self.computation_state))
    # end of aggregation_update()


    '''
    finish_aggregation_evaluate() - change to transient state
    '''
    def finish_aggregation_evaluate(self, cur_cycle):
        assert self.computation_state == 'aggregating'
        self.computation_state = 'aggregating-to-idle'
    # end of finish_aggregation_evalaute()

    '''
    finish_aggregation_update() - update the state and schedule dependent events
    '''
    def finish_aggregation_update(self, cur_cycle):
        assert self.computation_state == 'aggregating-to-idle'
        self.computation_state = 'idle'

        if self.communication_state == 'idle': # local aggregation
            if self.args.only_compute:
                return
            assert len(self.pending_aggregations) == 0
            self.communication_state = 'reduce-scatter'
            assert len(self.reduce_scatter_schedule) > 0
            if self.reduce_scatter_schedule[0] == None:
                self.schedule('reduce-scatter', cur_cycle + self.estimated_steptime)
                self.estimated_next_steptime = cur_cycle + self.estimated_steptime
                if self.id == 14:
                    logger.info('{} | {} | 0 schedule for None next time {}'.format(cur_cycle, self.name, self.estimated_next_steptime))
            else:
                self.schedule('reduce-scatter', cur_cycle + 1)

        elif len(self.pending_aggregations) > 0: # allreduce aggregation
            # clear dependency
            flow, child, _ = self.pending_aggregations.pop(0)
            logger.info('{} | {} | clear pending aggregation for flow {} from child HMC-{}'.format(cur_cycle, self.name, flow, child))
            level = None
            # clear dependency
            if len(self.reduce_scatter_schedule) > 0:
                flow_child = (flow, child)
                dependent_flow = None
                for i, schedules in enumerate(self.reduce_scatter_schedule):
                    if schedules == None:
                        continue
                    for fl, schedule in schedules.items():
                        if flow_child in schedule[1]:
                            level = i
                            dependent_flow = fl
                            break
                self.reduce_scatter_schedule[level][dependent_flow][1].remove(flow_child)
                if self.new_step == True and self.args.strict_schedule:
                    if len(self.free_nis) == self.args.radix and len(self.just_allocated_nis) == 0:
                        self.schedule('reduce-scatter', cur_cycle + 1)
                elif len(self.free_nis) - len(self.just_allocated_nis) > 0:
                    self.schedule('reduce-scatter', cur_cycle + 1)

        logger.info('{} | {} | finishes aggregation , computation state: {}, communication state: {}'.format(cur_cycle, self.name, self.computation_state, self.communication_state))

        if len(self.pending_aggregations) > 0:
            self.schedule('aggregation', cur_cycle + 1)
    # end of finish_aggregation_update()


    '''
    reduce_scatter_evaluate() - select scheduled communications, not started
    '''
    def reduce_scatter_evaluate(self, cur_cycle):
        assert self.communication_state == 'reduce-scatter'
        assert len(self.free_nis) > 0 or cur_cycle == self.estimated_next_steptime
        assert len(self.reduce_scatter_schedule) > 0
        assert len(self.just_allocated_nis) == 0
        if self.new_step == True and self.args.strict_schedule and \
                len(self.free_nis) != self.args.radix:
            if self.args.estimate_lockstep and len(self.reduce_scatter_schedule) > 1:
                if self.reduce_scatter_schedule[0] == None and cur_cycle == self.estimated_next_steptime:
                    self.reduce_scatter_schedule.pop(0)
                    if self.reduce_scatter_schedule[0] == None:
                        self.schedule('reduce-scatter', cur_cycle + self.estimated_steptime)
                        self.estimated_next_steptime = cur_cycle + self.estimated_steptime
            return
        self.new_step = False
        for ni in self.free_nis:
            assert self.messages_sent[ni] == 0
            send_flow = None
            parent = None
            dest_ni = None
            num_data_copy = None
            timestep = None
            if len(self.reduce_scatter_schedule) > 0:
                while self.reduce_scatter_schedule[0] == None:
                    if self.args.estimate_lockstep:
                        if cur_cycle == self.estimated_next_steptime:
                            self.reduce_scatter_schedule.pop(0)
                            if self.reduce_scatter_schedule[0] == None:
                                self.schedule('reduce-scatter', cur_cycle + self.estimated_steptime)
                                self.estimated_next_steptime = cur_cycle + self.estimated_steptime
                            else:
                                if self.new_step == True and self.args.strict_schedule:
                                    if len(self.free_nis) == self.args.radix and len(self.just_allocated_nis) == 0:
                                        self.schedule('reduce-scatter', cur_cycle + 1)
                                elif len(self.free_nis) - len(self.just_allocated_nis) > 0:
                                    self.schedule('reduce-scatter', cur_cycle + 1)
                        return
                    self.reduce_scatter_schedule.pop(0)
                for flow, schedule in self.reduce_scatter_schedule[0].items():
                    depending_children = schedule[1]
                    if len(depending_children) == 0:
                        send_flow = flow
                        parent = schedule[0][0]
                        dest_ni = schedule[0][1]
                        num_data_copy = schedule[2]
                        timestep = schedule[3]
                        break
            if send_flow != None:
                if self.args.oracle_lockstep:
                    if timestep != HMC.allreduce_timestep:
                        break
                    assert timestep == HMC.allreduce_timestep
                self.reduce_scatter_schedule[0].pop(send_flow)
                if parent != None:
                    assert dest_ni != None
                    assert self.sending[ni] == None
                    self.just_allocated_nis[ni] = (send_flow, parent, dest_ni, timestep)
                    if self.message_size == 0:
                        self.num_messages[ni] = 1
                        self.num_sub_messages[ni] = self.base_num_sub_messages * num_data_copy
                    else:
                        self.num_messages[ni] = self.base_num_messages * num_data_copy
                        self.num_sub_messages[ni] = self.base_num_sub_messages
                if len(self.reduce_scatter_schedule[0]) == 0:
                    self.reduce_scatter_schedule.pop(0)
                    self.new_step = True
                    if self.args.estimate_lockstep and len(self.reduce_scatter_schedule) > 0:
                        if self.reduce_scatter_schedule[0] == None:
                            self.schedule('reduce-scatter', cur_cycle + 2 * self.estimated_steptime)
                            self.estimated_next_steptime = cur_cycle + 2 * self.estimated_steptime
                        else:
                            if self.new_step == True and self.args.strict_schedule:
                                if len(self.free_nis) == self.args.radix and len(self.just_allocated_nis) == 0:
                                    self.schedule('reduce-scatter', cur_cycle + 1)
                            elif len(self.free_nis) - len(self.just_allocated_nis) > 0:
                                self.schedule('reduce-scatter', cur_cycle + 1)
                    break # finish one timestep, should retry later
            else:
                break
    # end of reduce_scatter_evaluate()

    '''
    reduce_scatter_update() - schedule selected communications
    '''
    def reduce_scatter_update(self, cur_cycle):
        assert self.communication_state == 'reduce-scatter'
        assert len(self.free_nis) > 0
        if len(self.just_allocated_nis) > 0:
            # allocate NIs
            for ni, new_flow in self.just_allocated_nis.items():
                if self.sending[ni] != None:
                    print('{} | {} | NI {} - sending {}'.format(cur_cycle, self.name, ni, self.sending))
                assert self.sending[ni] == None
                self.sending[ni] = new_flow
                self.free_nis.remove(ni)
                logger.info('{} | {} | start reducing for flow {} (from NI {}) to parent HMC-{} (to NI {}) at timestep {}'.format(cur_cycle, self.name, new_flow[0], ni, new_flow[1], new_flow[2], new_flow[3]))
            self.just_allocated_nis.clear()
            logger.debug('{} | {} | schedule send-reduce-message for next cycle (new flow)'.format(cur_cycle, self.name))
            self.schedule('send-reduce-message', cur_cycle + 1)
        elif len(self.reduce_scatter_schedule) == 0 and len(self.free_nis) == self.args.radix:
            if self.args.only_reduce_scatter:
                self.communication_state = 'idle'
                return
            self.communication_state = 'all-gather'
            self.schedule('all-gather', cur_cycle + 1)
            logger.info('{} | {} | start all-gather (start in reduce-scatter)'.format(cur_cycle, self.name))
    # end of reduce_scatter_update()


    '''
    send_reduce_message_update() - send reduce messages
    '''
    def send_reduce_message_evaluate(self, cur_cycle):
        assert self.communication_state == 'reduce-scatter'
        for ni, sending in enumerate(self.sending):
            if sending == None:
                continue
            if not self.to_network_message_buffers[ni].is_full():
                flow = sending[0]
                dest = sending[1]
                dest_ni = sending[2]
                priority = 0
                if self.args.prioritize_schedule:
                    priority = sending[3]
                src_node = self.id * self.args.radix + ni
                dest_node = dest * self.args.radix + dest_ni
                if self.num_sub_messages[ni] == 1:
                    submsgtype = pybooksim.Message.HeadTail
                elif self.messages_sent[ni] % self.num_sub_messages[ni] == 0:
                    submsgtype = pybooksim.Message.Head
                elif (self.messages_sent[ni] + 1) % self.num_sub_messages[ni] == 0:
                    submsgtype = pybooksim.Message.Tail
                else:
                    submsgtype = pybooksim.Message.Body
                self.messages_sent[ni] += 1
                self.total_messages_sent += 1
                end = False
                if self.messages_sent[ni] == self.num_messages[ni] * self.num_sub_messages[ni]:
                    end = True
                message = Message(flow, self.cur_mids[ni], src_node, dest_node, self.sub_message_size, pybooksim.Message.ReduceData, submsgtype, priority, end)
                self.to_network_message_buffers[ni].enqueue(message, cur_cycle, 1)
                #print('{} | {} | sends a reduce message to for flow {} (from NI {}) to parent HMC-{} (to NI {}), sent messages {}'.format(cur_cycle, self.name, flow, ni, dest, dest_ni, self.messages_sent[ni]))
    # end of send_reduce_message_evaluate()

    '''
    send_reduce_message_update() - try to schedule event to select remaining communications
    '''
    def send_reduce_message_update(self, cur_cycle):
        assert self.communication_state == 'reduce-scatter'
        num_complete = 0
        for ni, sending in enumerate(self.sending):
            if sending == None:
                continue
            if self.messages_sent[ni] % self.num_sub_messages[ni] == 0:
                self.cur_mids[ni] = HMC.cur_mid
                HMC.cur_mid = (HMC.cur_mid + 1) % 2147483647
            if self.messages_sent[ni] < self.num_messages[ni] * self.num_sub_messages[ni]:
                self.schedule('send-reduce-message', cur_cycle + 1)
            else:
                flow = sending[0]
                dest = sending[1]
                dest_ni = sending[2]
                timestep = sending[3]
                if self.args.oracle_lockstep:
                    HMC.allreduce_remaining_for_timestep[timestep] -= 1
                    if HMC.allreduce_remaining_for_timestep[timestep] == 0:
                        HMC.allreduce_timestep += 1
                        for hmc in HMC.hmcs:
                            assert hmc.communication_state == 'reduce-scatter'
                            hmc.schedule('reduce-scatter', cur_cycle + 1)
                logger.info('{} | {} | finishes reducing for flow {} (from NI {}) to parent HMC-{} (to NI {}) at timestep {}'.format(cur_cycle, self.name, flow, ni, dest, dest_ni, timestep))
                self.num_messages[ni] = None
                self.num_sub_messages[ni] = None
                self.messages_sent[ni] = 0
                self.sending[ni] = None
                self.free_nis.add(ni)
                num_complete += 1
        # TODO: figure out the following two cases
        #   1. not sure why num_complete check make it worse, some discrepancy
        #      in finish-aggregation for scheduling reduce-scatter and
        #      incoming-message for scheduling all-gather;
        #   2. strict-schedule also make it worse.
        #if num_complete > 0 and len(self.reduce_scatter_schedule) > 0:
        if len(self.reduce_scatter_schedule) > 0:
            if self.new_step == True and self.args.strict_schedule:
                if len(self.free_nis) == self.args.radix and len(self.just_allocated_nis) == 0:
                    self.schedule('reduce-scatter', cur_cycle + 1)
                    logger.debug('{} | {} | schedule event for reduce-scatter'.format(cur_cycle, self.name))
            elif len(self.free_nis) - len(self.just_allocated_nis) > 0:
                self.schedule('reduce-scatter', cur_cycle + 1)
                logger.debug('{} | {} | schedule event for reduce-scatter'.format(cur_cycle, self.name))
        #elif num_complete > 0 and len(self.reduce_scatter_schedule) == 0 and \
        elif len(self.reduce_scatter_schedule) == 0 and \
                len(self.free_nis) == self.args.radix and \
                len(self.just_allocated_nis) == 0:
            # finish all the reduce-scatter from this node before it starts all-gather
            if self.args.only_reduce_scatter:
                self.communication_state = 'idle'
                return
            self.communication_state = 'all-gather'
            self.schedule('all-gather', cur_cycle + 1)
            logger.info('{} | {} | start all-gather (start in send-reduce-message)'.format(cur_cycle, self.name))
    # end of send_reduce_message_update()


    '''
    all_gather_evaluate() - select scheduled communications, not started
    '''
    def all_gather_evaluate(self, cur_cycle):
        assert self.communication_state == 'all-gather'
        assert len(self.all_gather_schedule) > 0
        assert len(self.free_nis) > 0
        assert len(self.just_allocated_nis) == 0
        if self.new_step == True and self.args.strict_schedule and \
                len(self.free_nis) != self.args.radix:
            return
        self.new_step = False
        for ni in self.free_nis:
            assert self.messages_sent[ni] == 0
            assert self.sending[ni] == None
            send_flow = None
            num_data_copy = None
            timestep = None
            if len(self.all_gather_schedule) > 0:
                for flow, schedule in self.all_gather_schedule[0].items():
                    depending_parent = schedule[1]
                    if depending_parent == None:
                        send_flow = flow
                        num_data_copy = schedule[2]
                        timestep = schedule[3]
                        break
            if send_flow != None:
                child, dest_ni = self.all_gather_schedule[0][send_flow][0].pop(0)
                if len(self.all_gather_schedule[0][send_flow][0]) == 0:
                    self.all_gather_schedule[0].pop(send_flow)
                if len(self.all_gather_schedule[0]) == 0:
                    self.all_gather_schedule.pop(0)
                    self.new_step = True
                self.just_allocated_nis[ni] = (send_flow, child, dest_ni, timestep)
                if self.message_size == 0:
                    self.num_messages[ni] = 1
                    self.num_sub_messages[ni] = self.base_num_sub_messages * num_data_copy
                else:
                    self.num_messages[ni] = self.base_num_messages * num_data_copy
                    self.num_sub_messages[ni] = self.base_num_sub_messages
            else:
                break
    # end of all_gather_evaluate()


    '''
    all_gather_update() - schedule selected communications
    '''
    def all_gather_update(self, cur_cycle):
        assert self.communication_state == 'all-gather'
        assert len(self.free_nis) > 0
        if len(self.just_allocated_nis) > 0:
            # allocate NIs
            for ni, new_flow in self.just_allocated_nis.items():
                assert self.sending[ni] == None
                self.sending[ni] = new_flow
                self.free_nis.remove(ni)
                logger.info('{} | {} | start gathering for flow {} (from NI {}) to child HMC-{} (to NI {}) at timestep {}'.format(cur_cycle, self.name, new_flow[0], ni, new_flow[1], new_flow[2], new_flow[3]))
            self.just_allocated_nis.clear()
            logger.debug('{} | {} | schedule send-gather-message for next cycle (new flow)'.format(cur_cycle, self.name))
            self.schedule('send-gather-message', cur_cycle + 1)
    # end of all_gather_update()


    '''
    send_gather_message_evaluate() - send gather messages
    '''
    def send_gather_message_evaluate(self, cur_cycle):
        assert self.communication_state == 'all-gather'
        for ni, sending in enumerate(self.sending):
            if sending == None:
                continue
            if not self.to_network_message_buffers[ni].is_full():
                flow = sending[0]
                dest = sending[1]
                dest_ni = sending[2]
                timestep = sending[3]
                src_node = self.id * self.args.radix + ni
                dest_node = dest * self.args.radix + dest_ni
                if self.num_sub_messages[ni] == 1:
                    submsgtype = pybooksim.Message.HeadTail
                elif self.messages_sent[ni] % self.num_sub_messages[ni] == 0:
                    submsgtype = pybooksim.Message.Head
                elif (self.messages_sent[ni] + 1) % self.num_sub_messages[ni] == 0:
                    submsgtype = pybooksim.Message.Tail
                else:
                    submsgtype = pybooksim.Message.Body
                self.messages_sent[ni] += 1
                self.total_messages_sent += 1
                end = False
                if self.messages_sent[ni] == self.num_messages[ni] * self.num_sub_messages[ni]:
                    end = True
                message = Message(flow, self.cur_mids[ni], src_node, dest_node, self.sub_message_size, pybooksim.Message.GatherData, submsgtype, timestep, end)
                self.to_network_message_buffers[ni].enqueue(message, cur_cycle, 1)
                #print('{} | {} | sends a gather message to for flow {} (from NI {}) to child HMC-{} (to NI {}), sent messages {}'.format(cur_cycle, self.name, flow, ni, dest, dest_ni, self.messages_sent[ni]))
    # end of send_gather_message_evaluate()

    '''
    send_gather_message_update() - try to schedule event to select remaining communications
    '''
    def send_gather_message_update(self, cur_cycle):
        assert self.communication_state == 'all-gather'
        num_complete = 0
        for ni, sending in enumerate(self.sending):
            if sending == None:
                continue
            if self.messages_sent[ni] % self.num_sub_messages[ni] == 0:
                self.cur_mids[ni] = HMC.cur_mid
                HMC.cur_mid = (HMC.cur_mid + 1) % 2147483647
            if self.messages_sent[ni] < self.num_messages[ni] * self.num_sub_messages[ni]:
                self.schedule('send-gather-message', cur_cycle + 1)
            else:
                flow = sending[0]
                dest = sending[1]
                dest_ni = sending[2]
                timestep = sending[3]
                logger.info('{} | {} | finishes gather for flow {} (from NI {}) to child HMC-{} (to NI {}) at timestep {}'.format(cur_cycle, self.name, flow, ni, dest, dest_ni, timestep))
                self.num_messages[ni] = None
                self.num_sub_messages[ni] = None
                self.messages_sent[ni] = 0
                self.sending[ni] = None
                self.free_nis.add(ni)
                num_complete += 1
        #if num_complete > 0 and len(self.all_gather_schedule) > 0:
        if len(self.all_gather_schedule) > 0:
            if self.new_step == True and self.args.strict_schedule:
                if len(self.free_nis) == self.args.radix and len(self.just_allocated_nis) == 0:
                    self.schedule('all-gather', cur_cycle + 1)
                    logger.debug('{} | {} | schedule all-gather (more schedules to send in send-gather-message)'.format(cur_cycle, self.name))
            elif len(self.free_nis) - len(self.just_allocated_nis) > 0:
                self.schedule('all-gather', cur_cycle + 1)
                logger.debug('{} | {} | schedule all-gather (more schedules to send in send-gather-message)'.format(cur_cycle, self.name))
        #elif num_complete > 0 and len(self.all_gather_schedule) == 0 and \
        elif len(self.all_gather_schedule) == 0 and \
                len(self.free_nis) == self.args.radix and \
                len(self.just_allocated_nis) == 0:
            self.communication_state = 'idle'
            logger.info('{} | {} | finishes all-gather (after send-gather-message)'.format(cur_cycle, self.name))
    # end of send_gather_message_update()


    '''
    incoming_message_evaluate() - receive messages
    '''
    def incoming_message_evaluate(self, cur_cycle):
        for ni, message_buffer in enumerate(self.from_network_message_buffers):
            message = message_buffer.peek(cur_cycle)
            if message == None:
                continue
            self.from_network_message_buffers[ni].dequeue(cur_cycle)
            src = message.src // self.args.radix
            src_ni = message.src % self.args.radix
            if message.type == pybooksim.Message.ReduceData:
                self.messages_received['reduce-scatter'][message.flow][src] += 1
                if message.end:
                    self.messages_received_end['reduce-scatter'][message.flow][src] = True
                #print('{} | {} | receives a reduce message for flow {} (at NI {}) from child HMC-{} (from NI {}), received messages {}'.format(cur_cycle, self.name, message.flow, ni, src, src_ni, self.messages_received['reduce-scatter'][message.flow][src]))
            elif message.type == pybooksim.Message.GatherData:
                self.messages_received['all-gather'][message.flow] += 1
                if message.end:
                    self.messages_received_end['all-gather'][message.flow] = True
                #print('{} | {} | receives a gather message for flow {} (at NI {}) from parent HMC-{} (from NI {}), received messages {}'.format(cur_cycle, self.name, message.flow, ni, src, src_ni, self.messages_received['all-gather'][message.flow]))
    # end of incomming_message_evaluate()

    '''
    incoming_message_update() - check states and try to schedule event to select remaining communications
    '''
    def incoming_message_update(self, cur_cycle):
        for flow in range(self.args.num_hmcs):
            # handle reduce-scatter
            for src, messages_received_end in self.messages_received_end['reduce-scatter'][flow].items():
                if messages_received_end:
                    logger.info('{} | {} | receives full reduce for flow {} from child HMC-{}'.format(cur_cycle, self.name, flow, src))
                    if self.computation_state == 'idle':
                        self.schedule('aggregation', cur_cycle + 1)
                    # TODO: prioritize aggregation based on their timestep
                    self.pending_aggregations.append((flow, src, self.messages_received['reduce-scatter'][flow][src]))
                    self.messages_received['reduce-scatter'][flow][src] = 0
                    self.messages_received_end['reduce-scatter'][flow][src] = False

            # handle all-gather
            if self.messages_received_end['all-gather'][flow] == True:
                src = self.allreduce.trees_parent[flow][self.id]
                logger.info('{} | {} | receives full gather for flow {} from parent HMC-{}'.format(cur_cycle, self.name, flow, src))
                self.messages_received['all-gather'][flow] = 0
                self.messages_received_end['all-gather'][flow] = False
                if len(self.all_gather_schedule) > 0:
                    # clear all the dependencies
                    flow_parent = (flow, src)
                    for i, schedules in enumerate(self.all_gather_schedule):
                        for fl, schedule in schedules.items():
                            if flow_parent == schedules[fl][1]: # from depending flow-parent
                                children = self.all_gather_schedule[i][fl][0]
                                num_data_copy = self.all_gather_schedule[i][fl][2]
                                timestep = self.all_gather_schedule[i][fl][3]
                                self.all_gather_schedule[i][fl] = (children, None, num_data_copy, timestep)
                    if self.communication_state == 'all-gather':
                        if self.new_step == True and self.args.strict_schedule:
                            if len(self.free_nis) == self.args.radix and len(self.just_allocated_nis) == 0:
                                self.schedule('all-gather', cur_cycle + 1)
                                logger.debug('{} | {} | schedule all-gather (more schedules to send in incoming-message)'.format(cur_cycle, self.name))
                        elif len(self.free_nis) - len(self.just_allocated_nis) > 0:
                            self.schedule('all-gather', cur_cycle + 1)
                            logger.debug('{} | {} | schedule all-gather (more schedules to send in incoming-message)'.format(cur_cycle, self.name))
                elif len(self.free_nis) == self.args.radix and \
                        len(self.just_allocated_nis) == 0:
                    self.communication_state = 'idle'
                    logger.info('{} | {} | finishes all-gather (after recieving all-gather)'.format(cur_cycle, self.name))
    # end of incoming_message_update()


    '''
    aggregate() - aggregate all the weight updates of all the NPUs

    return: the cycles of aggregation
    '''
    def aggregate(self):
        if not self.pending_aggregations:
            if HMC.model_aggregation_cycles == None:
                partial_model_per_npu = math.ceil(self.model.size / self.num_npus)
                cycles = self.npu.aggregate(partial_model_per_npu, self.num_npus)
                HMC.model_aggregation_cycles = cycles

            self.compute_cycles += HMC.model_aggregation_cycles

            return HMC.model_aggregation_cycles

        else:
            # TODO: change me
            return 1
            flow, child, num_sub_messages = self.pending_aggregations[0]
            data_size = num_sub_messages * self.sub_message_size // self.bytes_per_param  # NO. params
            if data_size not in HMC.allreduce_aggregation_cycles.keys():
                partial_aggregation_per_npu = math.ceil(data_size / self.num_npus)
                cycles = self.npu.aggregate(partial_aggregation_per_npu, self.num_npus)
                HMC.allreduce_aggregation_cycles[data_size] = cycles

            self.allreduce_compute_cycles += HMC.allreduce_aggregation_cycles[data_size]

            return HMC.allreduce_aggregation_cycles[data_size]
    # end of aggregate()


    '''
    inference() - inference processing of the NN model

    return: number of cycles for parallel inference
    '''
    def inference(self):

        if HMC.inference_cycles == None:
            npu_cycles = self.npu.inference(self.model)
            cycles = npu_cycles * self.samples_per_npu
            HMC.inference_cycles = cycles

        self.compute_cycles += HMC.inference_cycles

        return HMC.inference_cycles
    # end of inference()


    '''
    train() - training of the NN model

    return: number of cycles for training
    '''
    def train(self):

        if HMC.training_cycles == None:
            npu_cycles = self.npu.train(self.model)
            cycles = npu_cycles * self.samples_per_npu
            HMC.training_cycles = cycles

        self.compute_cycles += HMC.training_cycles

        return HMC.training_cycles
    # end of train()

