from typing import Callable, Dict, Optional
from tensorflow.keras.optimizers import Adam

class Model:
    def __init__(self, dataset_cls: type, network_fn: Callable, 
                 dataset_args: Dict = None, network_args: Dict = None):
        self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'
        
        self.data = None
        self.network = None

    @property
    def image_shape(self):
        return self.data.input_shape

    @property
    def weights_filename(self) -> str:
        pass

    def fit(self, dataset, batch_size: int = 32, epochs: int = 10,
            augment_val: bool = True, callbacks: list = None):
        pass

    def evaluate(self, x, y, batch_size: int=16, verbose: bool=False):
        pass
    
    def loss(self):
        pass

    def optimizer(self):
        Adam()

    def metrics(self):
        pass

    def decode_output(self, x):
        return x

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)

        
        
