from torch.utils.data.dataset import Dataset
from torchvision import transforms as tr
from PIL import Image
import pandas as pd
import torch

class ImageDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 file_name: str
                 ) -> None:
        super().__init__()
        df = pd.read_pickle(f"{dataset_path}/{file_name}")

        self.dataset_path = dataset_path
        self.images = df['fname'].values.astype(str)
        self.labels = df['label'].values.astype(int)

        self.transform = tr.Compose([
            tr.Resize((224, 224)),
            tr.ToTensor(),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = f"{self.dataset_path}/images/{self.images[idx]}.png"

        image = Image.open(path)
        label = self.labels[idx]

        image = self.transform(image)

        return image, label