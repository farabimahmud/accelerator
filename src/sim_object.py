from abc import ABC, abstractmethod

class SimObject(ABC):
    def __init__(self, eventq):
        self.global_eventq = eventq


    '''
    process() - event processing function in a particular cycle
    @cur_cycle: the current cycle that with events to be processed
    '''
    @abstractmethod
    def process(self, cur_cycle):
        pass
