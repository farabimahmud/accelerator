import math

from sim_object import SimObject
from npu import NPU
from eventq import EventQueue
from message_buffer import *


class HMC(SimObject):
    # like static variables, reducing simulation time for data-parallel training
    inference_cycles = None
    training_cycles = None
    aggregation_cycles = None

    def __init__(self, i, args, eventq):
        super().__init__(eventq)
        self.id = i
        self.name = 'HMC-{}'.format(self.id)
        self.args = args
        self.npu = NPU(args)
        self.num_npus = self.args.num_vaults

        self.local_eventq = EventQueue()
        self.state = None

        self.compute_cycles = 0

        self.from_network_message_buffer = None
        self.to_network_message_buffer = None

        self.model = None
        self.bytes_per_param = 4 # bytes
        self.mini_batch_per_npu = math.ceil(self.args.mini_batch_size / self.args.num_vaults)

        self.message_size = 64 # bytes
        self.num_messages = None
        self.allreduce = None
        self.step = 0
        self.messages_sent = 0
        self.messages_received = 0


    '''
    load_model() - assign the NN model to this hmc
    @model: the NN model to be loaded
    '''
    def load_model(self, model):
        self.model = model
        self.num_messages = math.ceil(self.model.size * self.bytes_per_param / self.message_size)
    # end of load_model()


    '''
    startup() - startup function for simulation of HMC

    desc - schedule the start event for the simulation of HMC. Currently, assuming
           we are doing training only.
    TODO: should be extended later for more functionalities
    '''
    def startup(self):
        # currently start from training
        self.state = 'idle'
        self.local_eventq.schedule('training', 0)
        self.global_eventq.schedule(self, 0)
    # end of startup()


    '''
    set_allreduce() - set allreduce schedule
    @allreduce: allreduce schedule
    '''
    def set_allreduce(self, allreduce):
        self.allreduce = allreduce
    # end of set_allreduce()


    '''
    set_message_buffers() - set message buffers connected with network
    @from_network_message_buffer: message buffer for incoming messages
    @to_network_message_buffer: message buffer for outgoing messages
    '''
    def set_message_buffers(self, from_network_message_buffer, to_network_message_buffer):
        self.from_network_message_buffer = from_network_message_buffer
        self.to_network_message_buffer = to_network_message_buffer
    # end of set_message_buffers


    '''
    schedule() - schedule the event at a given time
    @event: the event to be scheduled
    @cycle: scheduled time
    '''
    def schedule(self, event, cycle):
        self.local_eventq.schedule(event, cycle)
        self.global_eventq.schedule(self, cycle)
    # end of reschedule()


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

        for event in events:
            if event == 'training':
                if self.state == 'idle':
                    self.state = 'training'
                    cycles = self.train()
                    self.schedule('finish-training', cur_cycle + cycles)
                    #print('HMC {} starts training at cycle {}, state: {}'.format(self.id, cur_cycle, self.state))
                else:
                    self.reschedule(event)

            elif event == 'finish-training':
                assert self.state == 'training'
                self.state = 'idle'
                self.schedule('aggregation', cur_cycle + 1)
                #print('HMC {} finishes training at cycle {}, sate: {}'.format(self.id, cur_cycle, self.state))

            elif event == 'aggregation':
                if self.state == 'idle':
                    self.state = 'aggregating'
                    cycles = self.aggregate()
                    self.schedule('finish-aggregation', cur_cycle + cycles)
                    #print('HMC {} starts aggregation at cycle {}, state: {}'.format(self.id, cur_cycle, self.state))
                else:
                    self.reschedule(event)

            elif event == 'finish-aggregation':
                assert self.state == 'aggregating'
                #self.state = 'idle'
                self.state = 'reduce'
                #print('HMC {} finishes aggregation at cycle {}, state: {}'.format(self.id, cur_cycle, self.state))
                if self.allreduce.reduce_sender_in_iteration(self.step, self.id):
                    self.schedule('reduce', cur_cycle + 1)

            elif event == 'reduce':
                self.state = 'reduce'
                assert self.messages_sent == 0
                self.schedule('send-reduce-message', cur_cycle + 1)
                #print('{} | {} | start reducing ...'.format(cur_cycle, self.name))

            elif event == 'gather':
                self.state = 'gather'
                assert self.messages_sent == 0
                if self.allreduce.broadcast_sender_in_iteration(self.step, self.id):
                    self.schedule('send-gather-message', cur_cycle + 1)

            elif event == 'send-reduce-message':
                dest = self.allreduce.get_reduce_dest(self.step, self.id)
                message = Message(self.id, dest, 64)
                self.to_network_message_buffer.enqueue(message, cur_cycle, 1)
                self.messages_sent += 1
                #print('{} | {} | Step {}: sends a reduce message to HMC {}'.format(cur_cycle, self.name, self.step, dest))
                if self.messages_sent < self.num_messages:
                    self.schedule('send-reduce-message', cur_cycle + 1)
                else:
                    self.messages_sent = 0
                    self.step = self.allreduce.get_iterations() - self.step - 1
                    self.state = 'gather'

            elif event == 'send-gather-message':
                dest = self.allreduce.get_broadcast_dest(self.step, self.id)
                message = Message(self.id, dest, 64)
                self.to_network_message_buffer.enqueue(message, cur_cycle, 1)
                self.messages_sent += 1
                if self.messages_sent < self.num_messages:
                    self.schedule('send-gather-message', cur_cycle + 1)
                else:
                    self.messages_sent = 0
                    if self.step + 1 < self.allreduce.get_iterations():
                        self.step += 1
                        self.schedule('gather', cur_cycle + 1)
                    else:
                        self.state = 'idle'
                        #print('{} | {} finishes gather at step {}'.format(cur_cycle, self.name, self.step))


            elif event == 'incoming-message':
                self.messages_received += 1
                message = self.from_network_message_buffer.peek(cur_cycle)
                self.from_network_message_buffer.dequeue(cur_cycle)
                assert message != None
                #if self.state == 'reduce':
                #    print('{} | {} | Step {}: receives a reduce messsage from HMC {}'.format(cur_cycle, self.name, self.step, message.src))
                #elif self.state == 'gather':
                #    print('{} | {} | Step {}: receives a gather messsage from HMC {}'.format(cur_cycle, self.name, self.step, message.src))
                if self.messages_received == self.num_messages:
                    self.messages_received = 0
                    if self.state == 'reduce':
                        assert self.allreduce.reduce_sender_in_iteration(self.step, message.src)
                        assert self.allreduce.get_reduce_dest(self.step, message.src) == self.id
                        if self.step + 1 < self.allreduce.get_iterations():
                            self.step += 1
                            self.state = 'idle'
                            self.schedule('aggregation', cur_cycle + 1)
                        else:
                            self.step = 0
                            self.schedule('gather', cur_cycle + 1)
                            #print('{} | {} | start gathering ...'.format(cur_cycle, self.name))
                    elif self.state == 'gather':
                        assert self.allreduce.broadcast_sender_in_iteration(self.step , message.src)
                        assert self.allreduce.get_broadcast_dest(self.step, message.src) == self.id
                        if self.step + 1 < self.allreduce.get_iterations():
                            self.step += 1
                            self.schedule('gather', cur_cycle + 1)
                            #print('{} | {} | start gathering ...'.format(cur_cycle, self.name))
                        else:
                            self.state = 'idle'

            else:
                raise RuntimeError('Unknown event type {} for {}'.format(event, self.name))
    # end of process()


    '''
    aggregate() - aggregate all the weight updates of all the NPUs

    return: the cycles of aggregation
    '''
    def aggregate(self):
        if HMC.aggregation_cycles == None:
            partial_model_per_npu = math.ceil(self.model.size / self.num_npus)
            cycles = self.npu.aggregate(partial_model_per_npu, self.num_npus)
            HMC.aggregation_cycles = cycles

        self.compute_cycles += HMC.aggregation_cycles

        return HMC.aggregation_cycles
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

