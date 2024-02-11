from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
import numpy as np

from typing import Tuple

class ThreeBodiesDataset:
    
    def __init__(self, dir: str = 'data/three_bodies_data_2D.pt', device: str = 'cpu') -> None:
        
        data = torch.flatten(torch.load(dir, map_location=device), -2)
        self.dataset = TensorDataset(data)

    def get_dataloaders(self, batch_size: int = 10, train_size: float = .9) -> Tuple[DataLoader, DataLoader]:
    
        train_dataset, test_dataset = random_split(self.dataset, lengths=[train_size, 1 - train_size])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        return train_dataloader, test_dataloader

if __name__ == '__main__':
    d = ThreeBodiesDataset()
    train, test = d.get_dataloaders()
    print(next(iter(train))[0].shape)