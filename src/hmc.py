import sys
import math
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
    allreduce_aggregation_cycles = None

    def __init__(self, i, args, eventq):
        super().__init__(eventq)
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
        self.mini_batch_per_npu = math.ceil(self.args.mini_batch_size / self.args.num_vaults)

        self.message_size = 64 # bytes
        self.num_messages = None

        # for the schedule semantics, refer to allreduce/allreduce.py
        self.allreduce = None
        self.reduce_scatter_schedule = None
        self.all_gather_schedule = None

        self.sending = [None for i in range(self.args.radix)]
        self.free_nis = set([i for i in range(self.args.radix)])
        self.pending_aggregations = []
        self.step = 0
        # the local accelerator can only control what to send but not what to receive
        self.messages_sent = [0] * self.args.radix
        self.messages_received = {'reduce-scatter': [{} for i in range(self.args.num_hmcs)],
                                  'all-gather': [0] * self.args.num_hmcs}


    '''
    load_model() - assign the NN model to this hmc
    @model: the NN model to be loaded
    '''
    def load_model(self, model):
        self.model = model
        self.num_messages = math.ceil(self.model.size * self.bytes_per_param /
                self.message_size / self.args.num_hmcs)
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
        if self.args.only_allreduce == False:
            self.computation_state = 'idle'
            self.local_eventq.schedule('training', 0)
        else:
            self.computation_state = 'aggregating'
            self.local_eventq.schedule('finish-aggregation', 0)
        self.global_eventq.schedule(self, 0)
    # end of startup()


    '''
    set_allreduce() - set allreduce schedule
    @allreduce: allreduce schedule
    '''
    def set_allreduce(self, allreduce):
        self.allreduce = allreduce
        self.reduce_scatter_schedule = deepcopy(allreduce.reduce_scatter_schedule[self.id])
        self.all_gather_schedule = deepcopy(allreduce.all_gather_schedule[self.id])
        for root in range(self.args.num_hmcs):
            for child in allreduce.trees_children[root][self.id]:
                self.messages_received['reduce-scatter'][root][child] = 0
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

        # NOTE: the events have some unpredictable sequences, may lead to corner cases.
        #       Therefore, forcing the sequence of processing. First process the events
        #       that may change states that later processed events may depend on.
        if 'finish-training' in events:
            assert self.computation_state == 'training'
            self.computation_state = 'idle'
            self.schedule('aggregation', cur_cycle + 1)
            logger.info('{} | {} finishes training, computation sate: {}, communication state: {}'.format(cur_cycle, self.name, self.computation_state, self.communication_state))
            events.remove('finish-training')

        if 'training' in events:
            if self.computation_state == 'idle':
                self.computation_state = 'training'
                cycles = self.train()
                self.schedule('finish-training', cur_cycle + cycles)
                logger.info('{} | {} | starts training, computation state: {}, communication state: {}'.format(cur_cycle, self.name, self.computation_state, self.communication_state))
            else:
                self.reschedule(event)
            events.remove('training')

        if 'finish-aggregation' in events:
            assert self.computation_state == 'aggregating'
            self.computation_state = 'idle'
            if self.communication_state == 'idle':
                # local aggregation
                assert len(self.pending_aggregations) == 0
                self.communication_state = 'reduce-scatter'
                if len(self.reduce_scatter_schedule) > 0:
                    self.schedule('reduce-scatter', cur_cycle + 1)

            if len(self.pending_aggregations) > 0:
                # clear dependency
                flow, child = self.pending_aggregations.pop(0)
                logger.info('{} | {} | clear pending aggregation for flow {} from child HMC-{}'.format(cur_cycle, self.name, flow, child))
                level = None
                if len(self.reduce_scatter_schedule) > 0:
                    for i, schedules in enumerate(self.reduce_scatter_schedule):
                        if flow in schedules.keys():
                            assert child in schedules[flow][1]
                            level = i
                            break
                    self.reduce_scatter_schedule[level][flow][1].remove(child)
                    if len(self.free_nis) > 0:
                        self.schedule('reduce-scatter', cur_cycle + 1)

            logger.info('{} | {} | finishes aggregation , computation state: {}, communication state: {}'.format(cur_cycle, self.name, self.computation_state, self.communication_state))

            if len(self.pending_aggregations) > 0:
                self.schedule('aggregation', cur_cycle + 1)

            events.remove('finish-aggregation')

        if 'aggregation' in events:
            if self.computation_state == 'idle':
                self.computation_state = 'aggregating'
                cycles = self.aggregate()
                self.schedule('finish-aggregation', cur_cycle + cycles)
                logger.info('{} | {} | starts aggregation, computation state: {}, communication state: {}'.format(cur_cycle, self.name, self.computation_state, self.communication_state))
            else:
                self.reschedule(event)
            events.remove('aggregation')

        if 'reduce-scatter' in events:
            assert self.communication_state == 'reduce-scatter'
            assert len(self.free_nis) > 0
            assert self.reduce_scatter_schedule
            allocated_nis = []
            # schedule to send out ready reduce-scatter to available NIs
            for ni in self.free_nis:
                assert self.messages_sent[ni] == 0
                send_flow = None
                parent = None
                dest_ni = None
                if len(self.all_gather_schedule) > 0:
                    for flow, schedule in self.reduce_scatter_schedule[0].items():
                        depending_children = schedule[1]
                        if len(depending_children) == 0:
                            send_flow = flow
                            parent = schedule[0][0]
                            dest_ni = schedule[0][1]
                            break
                if send_flow != None:
                    self.reduce_scatter_schedule[0].pop(send_flow)
                    if len(self.reduce_scatter_schedule[0]) == 0:
                        self.reduce_scatter_schedule.pop(0)
                    if parent != None:
                        assert dest_ni != None
                        assert self.sending[ni] == None
                        self.sending[ni] = (send_flow, parent, dest_ni)
                        allocated_nis.append(ni)
                        logger.info('{} | {} | start reducing for flow {} (from NI {}) to parent HMC-{} (to NI {})'.format(cur_cycle, self.name, send_flow, ni, parent, dest_ni))
                    else:
                        if len(self.reduce_scatter_schedule) == 0 and len(self.free_nis) == self.args.radix:
                            self.communication_state = 'all-gather'
                            self.schedule('all-gather', cur_cycle + 1)
                            logger.info('{} | {} | schedule all-gather'.format(cur_cycle, self.name))
                        break
                else:
                    break
            # allocate NIs
            for ni in allocated_nis:
                self.free_nis.remove(ni)
            if len(allocated_nis) > 0:
                self.schedule('send-reduce-message', cur_cycle + 1)
            events.remove('reduce-scatter')

        if 'send-reduce-message' in events:
            assert self.communication_state == 'reduce-scatter'
            for ni, sending in enumerate(self.sending):
                if sending == None:
                    continue
                flow = sending[0]
                dest = sending[1]
                dest_ni = sending[2]
                src_node = self.id * self.args.radix + ni
                dest_node = dest * self.args.radix + dest_ni
                message = Message(flow, src_node, dest_node, self.message_size, pybooksim.Message.ReduceData)
                self.to_network_message_buffers[ni].enqueue(message, cur_cycle, 1)
                self.messages_sent[ni] += 1
                #print('{} | {} | sends a reduce message to for flow {} (from NI {}) to parent HMC-{} (to NI {})'.format(cur_cycle, self.name, flow, ni, dest, dest_ni))
                if self.messages_sent[ni] < self.num_messages:
                    self.schedule('send-reduce-message', cur_cycle + 1)
                else:
                    logger.info('{} | {} | finishes reducing for flow {} (from NI {}) to parent HMC-{} (to NI {})'.format(cur_cycle, self.name, flow, ni, dest, dest_ni))
                    self.messages_sent[ni] = 0
                    self.sending[ni] = None
                    self.free_nis.add(ni)
                    if len(self.reduce_scatter_schedule) > 0:
                        self.schedule('reduce-scatter', cur_cycle + 1)
                    elif len(self.free_nis) == self.args.radix:
                        self.communication_state = 'all-gather'
                        self.schedule('all-gather', cur_cycle + 1)
                        logger.info('{} | {} | schedule all-gather'.format(cur_cycle, self.name))
                        break
            events.remove('send-reduce-message')

        if 'all-gather' in events:
            assert self.communication_state == 'all-gather'
            assert len(self.all_gather_schedule) > 0
            assert len(self.free_nis) > 0
            allocated_nis = []
            for ni in self.free_nis:
                assert self.messages_sent[ni] == 0
                assert self.sending[ni] == None
                send_flow = None
                if len(self.all_gather_schedule) > 0:
                    for flow, schedule in self.all_gather_schedule[0].items():
                        depending_parent = schedule[1]
                        if depending_parent == None:
                            send_flow = flow
                            break
                if send_flow != None:
                    child, dest_ni = self.all_gather_schedule[0][send_flow][0].pop(0)
                    if len(self.all_gather_schedule[0][send_flow][0]) == 0:
                        self.all_gather_schedule[0].pop(send_flow)
                    if len(self.all_gather_schedule[0]) == 0:
                        self.all_gather_schedule.pop(0)
                    self.sending[ni] = (send_flow, child, dest_ni)
                    allocated_nis.append(ni)
                    logger.info('{} | {} | start gathering for flow {} (from NI {}) to child HMC-{} (to NI {})'.format(cur_cycle, self.name, send_flow, ni, child, dest_ni))
                else:
                    break
            # allocate NIs
            for ni in allocated_nis:
                self.free_nis.remove(ni)
            if len(allocated_nis) > 0:
                self.schedule('send-gather-message', cur_cycle + 1)
            events.remove('all-gather')

        if 'send-gather-message' in events:
            assert self.communication_state == 'all-gather'
            for ni, sending in enumerate(self.sending):
                if sending == None:
                    continue
                flow = sending[0]
                dest = sending[1]
                dest_ni = sending[2]
                src_node = self.id * self.args.radix + ni
                dest_node = dest * self.args.radix + dest_ni
                message = Message(flow, src_node, dest_node, self.message_size, pybooksim.Message.GatherData)
                self.to_network_message_buffers[ni].enqueue(message, cur_cycle, 1)
                self.messages_sent[ni] += 1
                #print('{} | {} | sends a gather message to for flow {} (from NI {}) to child HMC-{} (to NI {})'.format(cur_cycle, self.name, flow, ni, dest, dest_ni))
                if self.messages_sent[ni] < self.num_messages:
                    self.schedule('send-gather-message', cur_cycle + 1)
                else:
                    self.messages_sent[ni] = 0
                    self.sending[ni] = None
                    self.free_nis.add(ni)
                    if len(self.all_gather_schedule) > 0:
                        self.schedule('all-gather', cur_cycle + 1)
                    else:
                        self.communication_state = 'idle'
                        logger.info('{} | {} finishes all-gather'.format(cur_cycle, self.name))
            events.remove('send-gather-message')

        if 'incoming-message' in events:
            for ni, message_buffer in enumerate(self.from_network_message_buffers):
                message = message_buffer.peek(cur_cycle)
                if message == None:
                    continue
                self.from_network_message_buffers[ni].dequeue(cur_cycle)
                src = message.src // self.args.radix
                src_ni = message.src % self.args.radix
                if message.type == pybooksim.Message.ReduceData:
                    #print('{} | {} | receives a reduce messsage for flow {} (at NI {}) from child HMC-{} (from NI {})'.format(cur_cycle, self.name, message.flow, ni, src, src_ni))
                    self.messages_received['reduce-scatter'][message.flow][src] += 1
                    if self.messages_received['reduce-scatter'][message.flow][src] == self.num_messages:
                        logger.info('{} | {} | receives full reduce for flow {} (at NI {}) from child HMC-{} (from NI {})'.format(cur_cycle, self.name, message.flow, ni, src, src_ni))
                        self.messages_received['reduce-scatter'][message.flow][src] = 0
                        if self.computation_state == 'idle':
                            self.schedule('aggregation', cur_cycle + 1)
                        self.pending_aggregations.append((message.flow, src))
                elif message.type == pybooksim.Message.GatherData:
                    self.messages_received['all-gather'][message.flow] += 1
                    #print('{} | {} | receives a gather messsage for flow {} (at NI {}) from parent HMC-{} (from NI {})'.format(cur_cycle, self.name, message.flow, ni, src, src_ni))
                    # clear all the dependencies
                    if self.messages_received['all-gather'][message.flow] == self.num_messages:
                        self.messages_received['all-gather'][message.flow] = 0
                        logger.info('{} | {} | receives full gather for flow {} (at NI {}) from parent HMC-{} (from NI {})'.format(cur_cycle, self.name, message.flow, ni, src, src_ni))
                        if len(self.all_gather_schedule) > 0:
                            for i, schedules in enumerate(self.all_gather_schedule):
                                if message.flow in schedules.keys():
                                    if src == schedules[message.flow][1]: # from depending parent
                                        children = self.all_gather_schedule[i][message.flow][0]
                                        self.all_gather_schedule[i][message.flow] = (children, None)
                            if self.communication_state == 'all-gather' and len(self.free_nis) > 0:
                                self.schedule('all-gather', cur_cycle + 1)
                            #if len(self.free_nis) > 0 and len(self.reduce_scatter_schedule) == 0:
                            #    self.communication_state = 'all-gather'
                            #    self.schedule('all-gather', cur_cycle + 1)
                        else:
                            self.state = 'idle'
                            logger.info('{} | {} finishes all-gather'.format(cur_cycle, self.name))
            events.remove('incoming-message')

        if len(events) != 0:
            raise RuntimeError('Unknown event type {} for {}'.format(event, self.name))
    # end of process()


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
            if HMC.allreduce_aggregation_cycles == None:
                partial_aggregation_per_npu = math.ceil(self.model.size /
                        self.args.num_hmcs / self.num_npus)
                cycles = self.npu.aggregate(partial_aggregation_per_npu, self.num_npus)
                HMC.allreduce_aggregation_cycles = cycles

            self.allreduce_compute_cycles += HMC.allreduce_aggregation_cycles

            return HMC.allreduce_aggregation_cycles
    # end of aggregate()


    '''
    inference() - inference processing of the NN model

    return: number of cycles for parallel inference
    '''
    def inference(self):

        if HMC.inference_cycles == None:
            npu_cycles = self.npu.inference(self.model)
            cycles = npu_cycles * self.mini_batch_per_npu
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
            cycles = npu_cycles * self.mini_batch_per_npu
            HMC.training_cycles = cycles

        self.compute_cycles += HMC.training_cycles

        return HMC.training_cycles
    # end of train()

