from abc import ABC, abstractmethod


class Base(ABC):
    
    @abstractmethod
    def generate(self, inputs):
        pass
    
    