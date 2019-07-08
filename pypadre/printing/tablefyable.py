from abc import abstractmethod, ABCMeta


class Tablefyable(ABCMeta):

    @abstractmethod
    def tablefy_to_row(self):
        pass

    @staticmethod
    @abstractmethod
    def tablefy_header():
        pass