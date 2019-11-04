from collections import defaultdict


class Message:
    def __init__(self, src, dest, size):
        self.src = src
        self.dest = dest
        self.size = size


class MessageBuffer:
    def __init__(self, max_size=0):
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

