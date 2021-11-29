import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

# Imageset class that extends Dataset class to enable the use of Dataloader on our dataset
class Imageset(Dataset):
    def __init__(self, training_data, transform = transforms.Compose([transforms.ToTensor(), ])):
        self.training_data = training_data
        self.transform = transform

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('Slicing is Not Supported')                
        return (self.transform(self.training_data[key][0]), torch.tensor(self.training_data[key][1]))
            
    def __len__(self):
        return len(self.training_data)