from torch.utils import data
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    '''
    
    '''
    def __init__(self, cfg) -> None:
        '''
        
        '''
        self.cfg = cfg
        
        
    @abstractmethod
    def __len__(self, index):
        '''
        
        '''
        return 0
    
    
    @abstractmethod
    def __getitem__(self, index):
        '''
        '''
        pass