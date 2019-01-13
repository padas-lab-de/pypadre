from tests.proof_of_concept.eventlogging.eventlogger import ee

print('Main function. Events will be transmitted from here')
# emit
ee.emit("myevent", "testing with a string")
# -> "handler1 called with foo"

ee.emit("myotherevent", "bar")
# -> "handler2 called with bar"
