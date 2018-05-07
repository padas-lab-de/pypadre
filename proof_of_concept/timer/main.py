from timer import TimeKeeper
from TimeTester import TimeTester
def main():
    # Using the argument Timekeeper.LOW_PRIORITY will log all the messages
    # that have a LOW_PRIORITY and above
    # TimeKeeper.MED_PRIORITY will log only messages that have a medium priority
    # or above.
    # Logging can be turned off by using TimeKeeper.NO_LOGGING
    timer_obj = TimeKeeper(TimeKeeper.LOW_PRIORITY)
    timer_obj.log_timer("test1",TimeKeeper.HIGH_PRIORITY,"Test 1 timer")
    timer_obj.log_timer("test2",TimeKeeper.MED_PRIORITY, "Test 2 Timer")
    timer_obj.log_timer("test3",TimeKeeper.LOW_PRIORITY, "Test 3 timer")
    timer_obj.log_timer("test1")
    timer_obj.log_timer("test3")
    timer_obj.log_timer("test2")

    # Creating an object and maintaining a single TimeKeeper instance
    # throughout the life of the object
    tester_obj = TimeTester(TimeKeeper.MED_PRIORITY)
    tester_obj.function4()

    # Added a last timer object to demonstrate that if any timers are left in the
    # dictionary when the destructor is called, it would show an error
    timer_obj.log_timer("test_error1")
    timer_obj.log_timer("test_error2")


if __name__ == '__main__':
    main()