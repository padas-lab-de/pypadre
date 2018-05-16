from timer import TimeKeeper


class TimeTester:
    def __init__(self, timer_priority):
        self._timekeeper = TimeKeeper(timer_priority)

    def function1(self):
        self._timekeeper.log_timer("function1Timer",
                                   TimeKeeper.HIGH_PRIORITY,
                                   "Function1 Timer")
        # Code to be executed
        self._timekeeper.log_timer("function1Timer")

    def function2(self):
        self._timekeeper.log_timer("function2Timer",
                                   TimeKeeper.MED_PRIORITY,
                                   "Function2 Timer")
        # Code to be executed
        self._timekeeper.log_timer("function2Timer")

    def function3(self):
        self._timekeeper.log_timer("function3Timer",
                                   TimeKeeper.LOW_PRIORITY,
                                   "Function3 Timer")
        # Code to be executed
        self._timekeeper.log_timer("function3Timer")

    def function4(self):
        self._timekeeper.log_timer("function4Timer",
                                   TimeKeeper.HIGH_PRIORITY,
                                   "Function4 Timer")
        self.function2()
        self.function1()
        self.function3()
        self._timekeeper.log_timer("function4Timer")