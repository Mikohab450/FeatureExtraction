import abc
from DataPrep import create_annotations
class NetworkArchitecture(metaclass=abc.ABCMeta):
    '''
        abstract base class that represents architecture of the network (R-CNN, fast R-CNN etc.)
    '''


    def __init__(self):
        pass

    @abc.abstractmethod
    def extract_features_from_image(self):
        pass
    @abc.abstractmethod
    def train(self):
        pass


