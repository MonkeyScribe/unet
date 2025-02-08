import numpy as np
import torch
import random
from typing import Optional

class UnetData():

    def __init__(self, input_size : int, output_size : int, noize_scale : float, min_radius : int, 
                 max_radius : int, min_center : Optional[int]=None, max_center : Optional[int]=None):
        self.input_size = input_size
        self.noize_scale = noize_scale
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.output_size = output_size
        self.min_center = 0 if min_center is None else min_center
        self.max_center = self.input_size if max_center is None else max_center
    
    def generate(self, center : Optional[np.ndarray] = None, radius : Optional[np.float32] = None):
    
        input_size = self.input_size
        if center is None : 
            center =  np.random.randint(self.min_center, self.max_center, size=2)
        if radius is None:
            radius = random.randrange(self.min_radius,self.max_radius)
        row, col = np.indices((input_size, input_size))

        label = (pow(row -center[0],2) + pow(col -center[1],2) < radius**2).astype(int)
        
        noize = np.abs(np.random.normal(0, scale=0.25, size = (input_size, input_size)))
        data = np.clip(noize+label,a_min = 0,  a_max = 1)

        top = int((input_size - self.output_size)/2)
        label = label[top:top+self.output_size,top:top+self.output_size] 

        return torch.from_numpy(data).to(dtype=torch.float32),  torch.from_numpy(label).to(dtype=torch.float32)
    
    def generateBatch(self, n : int, seed : int = 42):

        np.random.seed(seed)

        ds = [self.generate() for _ in range(n)]

        data = [d for [d, _]in ds]
        labels = [l for [_,l] in ds]

        return torch.stack(data, dim = 0).unsqueeze(1), torch.stack(labels, dim = 0).unsqueeze(1)

