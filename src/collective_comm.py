
class CollectiveCommunication:
    def __init__(self, args):
        self.args = args
        self.iteration = 0

    def get_dest(self, iteration, src):
        pass

    def get_pairs(self, iteration):
        pass

    def get_iterations(self):
        return self.iterations

    def sender_in_iteration(self, iteration, src):
        pass


class TreeCC(CollectiveCommunication):
    def __init__(self, args):
        self.iterations = 4
        self.schedule = [
                {0: 1, 3: 2, 4: 5, 7: 6, 8: 9, 11: 10, 12: 13, 15: 14},
                {1: 2, 5: 6, 9: 10, 13: 14},
                {2: 6, 14: 10},
                {10: 6}
                ]

    def get_dest(self, iteration, src):
        dest = -1

        if src in self.schedule[iteration]:
            dest = self.schedule[iteration][src]

        return dest

    def get_pairs(self, iteration):
        return self.schedule[iteration]

    def sender_in_iteration(self, iteration, src):
        if src in self.schedule[iteration]:
            return True
        else:
            return False


class RingCC(CollectiveCommunication):
    def __init__(self, args):
        self.iterations = 16
        self.schedule = {0: 4, 1: 0, 2: 6, 3: 2, 4: 8, 5: 1, 6: 10, 7: 3,
                8: 12, 9: 5, 10: 9, 11: 7, 12: 13, 13: 14, 14: 15, 15: 11}

    def get_dest(self, iteration, src):
        return self.schedule[src]

    def get_pairs(self, iteration):
        return self.schedule

    def sender_in_iteration(self, iteration, src):
        if src in self.schedule:
            return True
        else:
            return False

