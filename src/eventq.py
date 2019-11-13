from collections import defaultdict


class EventQueue:
    def __init__(self):
        self.event_queue = defaultdict(set)
        self.cycles = 0


    '''
    schedule() - schedule the event at timestamp when
    @event: the event to be added to the event queue
    @when: the timestamp the event should be scheduled
    '''
    def schedule(self, event, when):
        self.event_queue[when].add(event)
    # end of schedule()


    '''
    get_events() - get the events scheduled at a timestamp when
    @when: the timestamp of query for events
    '''
    def get_events(self, when):
        assert when in self.event_queue.keys()
        events = self.event_queue.pop(when)

        if when > self.cycles:
            self.cycles = when

        return events
    # def get_events(self, when)


    '''
    next_event_cycle() - get the next earliest timestamp for scheudling

    return: the next earliest timestamp that has events to be scheduled
    '''
    def next_event_cycle(self):
        when = None
        if len(self.event_queue) > 0:
            when = min(self.event_queue.keys())

        return when
    # end of next_event_cycle()


    '''
    next_events() - get the next earliest scheduled events and timestamp

    return:
    @when: the next earliest timestamp that has events to be scheduled
    @events: the next events to be scheduled
    '''
    def next_events(self):
        when = min(self.event_queue.keys())
        events = self.event_queue.pop(when)

        if when > self.cycles:
            self.cycles = when

        return when, events
    # end of next_events()


    '''
    empty() - check whether the event queue is empty or not

    return: True if empty, otherwise False
    '''
    def empty(self):
        if self.event_queue:
            return False
        else:
            return True
    # end of empty()
