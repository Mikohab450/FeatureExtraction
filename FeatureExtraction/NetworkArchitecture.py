import abc

class NetworkArchitecture(metaclass=abc.ABCMeta):
    '''
        abstract base class that represents 
    '''
    def __init__(self):
        pass


    @abc.abstractmethod
    def extract_features_from_image(self):
        pass
    @abc.abstractmethod
    def train(self):
        pass


