import abc
from DataPrep import create_annotations

class NetworkArchitecture(metaclass=abc.ABCMeta):
    '''
        abstract base class that represents architecture of the network (R-CNN, fast R-CNN etc.)
    '''
    CNN_model=None
    list_of_models=[]
    def __init__(self):
        pass
    @abc.abstractmethod
    def load_model(self,path):
        pass
    @abc.abstractmethod
    def choose_model(self,idx):
        pass
    @abc.abstractmethod
    def extract_features_from_image(self,img_dir,classes,IoU_cutoff_object,IoU_cutoff_not_object):
        pass


