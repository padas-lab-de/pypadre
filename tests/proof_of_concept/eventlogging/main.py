'''
from tests.proof_of_concept.eventlogging.eventlogger import ee
print('Main function. Events will be transmitted from here')
# emit
ee.emit("myevent", "testing with a string")
# -> "handler1 called with foo"

ee.emit("myotherevent", "bar")
# -> "handler2 called with bar"
'''
from padre.eventhandler import eventemitter, process_events
event_dict = dict()
event_dict['EVENT_NAME'] = 'EVENT_LOG'
event_dict['args'] = 'HELLO WORLD'
eventemitter.emit('EVENT', event_dict)
process_events()