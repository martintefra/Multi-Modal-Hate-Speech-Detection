
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms


def preprocess_image(tweet_id):
    image_path = f'archive/img_resized/{tweet_id}.jpg'
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    return image

class MultimodalDataset(Dataset):
    """
    Multimodal dataset for images and text classification.
    """
    def __init__(self, csv_file, processor):
        self.data_frame = pd.read_csv(csv_file)
        self.processor = processor
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        tweet_id = self.data_frame.iloc[idx, 3]
        image = preprocess_image(tweet_id)
        text = self.data_frame.iloc[idx, 2]
        label = self.data_frame.iloc[idx, 4]
        
        encoding = self.processor(image, text, return_tensors='pt', truncation=True, max_length=40)
        for k,v in encoding.items():
            encoding[k] = v.squeeze()

            
        encoding["labels"] = torch.tensor(label)
    
        return encoding
