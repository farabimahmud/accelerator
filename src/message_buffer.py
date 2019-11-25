from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class Message:
    def __init__(self, flow, mid, src, dest, size, msgtype, submsgtype):
        self.flow = flow
        self.id = mid
        self.src = src
        self.dest = dest
        self.size = size
        self.type = msgtype
        self.submsgtype = submsgtype


class MessageBuffer:
    def __init__(self, name, max_size=0):
        self.name = name
        self.size = 0
        self.max_size = max_size
        self.message_queue = defaultdict(list)
        self.peek_cycle = None
        self.consumer = None


    def set_consumer(self, consumer):
        self.consumer = consumer


    def is_full(self):
        if self.max_size == 0:
            return False
        elif self.max_size == self.size:
            return True
        else:
            return False


    def are_n_slots_available(self, n):
        # infinite size
        if self.max_size == 0:
            return True
        else:
            if self.max_size - self.size >= n:
                return True
            else:
                return False


    def peek(self, cur_cycle):
        message = None
        for ready_cycle in sorted(self.message_queue.keys()):
            if ready_cycle > cur_cycle:
                break
            if len(self.message_queue[ready_cycle]) != 1:
                logger.debug('Message Buffer {} has {} messages (not exactly 1) at cycle {}'.format(self.name, len(self.message_queue[ready_cycle]), ready_cycle))
                for i, message in enumerate(self.message_queue[cur_cycle]):
                    logger.debug('message {} ({}): {}'.format(i, message, message.__dict__))
                raise RuntimeError('Message Buffer {} has {} messages (not exactly 1) at cycle {}'.format(self.name, len(self.message_queue[ready_cycle]),ready_cycle))
            assert len(self.message_queue[ready_cycle]) == 1
            self.peek_cycle = ready_cycle
            message = self.message_queue[ready_cycle][0]
            break

        return message


    def enqueue(self, message, cur_cycle, delta):
        arrival_cycle = cur_cycle + delta

        self.message_queue[arrival_cycle].append(message)
        self.size += 1
        # TODO: need to differentiate request/response to break protocol deadlock
        self.consumer.schedule('incoming-message', arrival_cycle)


    def dequeue(self, cur_cycle):
        self.message_queue.pop(self.peek_cycle)
        self.peek_cycle = None
        self.size -= 1

