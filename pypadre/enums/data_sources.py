from enum import Enum, unique


@unique
class DataSources(Enum):
    oml = "oml"
    sklearn = "sklearn"
