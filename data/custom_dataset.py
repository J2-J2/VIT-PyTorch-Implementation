from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd
import torch

IMAGENETTE_CLASSES = {
    'n01440764': 'tench',       
    'n02102040': 'English springer',
    'n02979186': 'cassette player', 
    'n03000684': 'chain saw',        
    'n03028079': 'church',         
    'n03394916': 'French horn',    
    'n03417042': 'garbage truck',    
    'n03425413': 'gas pump',        
    'n03445777': 'golf ball',       
    'n03888257' : 'parachute'        
}


class Imagenette2Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.class_names = [IMAGENETTE_CLASSES[cls] for cls in self.classes]

        self.img_paths = list(self.root_dir.rglob("*.JPEG"))
        self.img_paths.sort()

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        cls_name = img_path.parent.name
        label = self.class_to_idx[cls_name]

        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


