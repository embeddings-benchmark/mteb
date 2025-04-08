import json

from torch.utils.data import Dataset


class EmptyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Optionally apply any transformations
        if self.transform:
            item = self.transform(item)
        
        return item


class JSONLDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        self.data = []
        
        # Load data from JSONL file
        if isinstance(file_path, str):
            with open(file_path, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
        elif isinstance(file_path, list):
            for path in file_path:
                with open(path, 'r') as f:
                    for line in f:
                        self.data.append(json.loads(line))
        else:
            raise ValueError("file_path must be a string or a list of strings.")
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Optionally apply any transformations
        if self.transform:
            item = self.transform(item)
        
        return item