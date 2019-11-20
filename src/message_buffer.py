from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class Message:
    def __init__(self, flow, src, dest, size, msgtype):
        self.flow = flow
        self.src = src
        self.dest = dest
        self.size = size
        self.type = msgtype


class MessageBuffer:
    def __init__(self, name, max_size=0):
        self.name = name
        self.max_size = max_size
        self.message_queue = defaultdict(list)
        self.consumer = None


    def set_consumer(self, consumer):
        self.consumer = consumer


    def are_n_slots_available(self, n, cur_cycle):
        # infinite size
        if self.max_size == 0:
            return True
        else:
            raise RuntimeError('Finiate message buffer size: ' + self.max_size)
            return False


    def peek(self, cur_cycle):
        message = None
        if cur_cycle in self.message_queue.keys():
            if len(self.message_queue[cur_cycle]) != 1:
                logger.debug('Message Buffer {} has {} messages (not exactly 1) at cycle {}'.format(self.name, len(self.message_queue[cur_cycle]), cur_cycle))
                for i, message in enumerate(self.message_queue[cur_cycle]):
                    logger.debug('message {} ({}): {}'.format(i, message, message.__dict__))
                raise RuntimeError('Message Buffer {} has {} messages (not exactly 1) at cycle {}'.format(self.name, len(self.message_queue[cur_cycle]), cur_cycle))
            assert len(self.message_queue[cur_cycle]) == 1
            message = self.message_queue[cur_cycle][0]

        return message


    def enqueue(self, message, cur_cycle, delta):
        arrival_cycle = cur_cycle + delta

        self.message_queue[arrival_cycle].append(message)
        # TODO: need to differentiate request/response to break protocol deadlock
        self.consumer.schedule('incoming-message', arrival_cycle)


    def dequeue(self, cur_cycle):
        self.message_queue.pop(cur_cycle)

