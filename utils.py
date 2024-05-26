import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import ViltProcessor, ViltModel

from models import MultimodalDataset


def collate_fn(batch, processor):
    """
    Collate function to pad batch data.

    Args:
        batch (list): List of dictionaries where each dictionary represents a sample.
        processor (ViltProcessor): Processor used to encode the image and text.
        
    Returns:
        dict: Dictionary containing the padded batch data.
    """
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # create padded pixel values and corresponding pixel mask
    encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")
    
    # create new batch
    batch = {}
    batch['input_ids'] = pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    batch['attention_mask'] = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    batch['token_type_ids'] = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = torch.tensor(labels)

    return batch


def inference(dataset_path, device="mps", batch_size=16):
    """
    Function to perform inference on a dataset.

    Args:
        dataset (MultimodalDataset): Dataset to perform inference on.
        processor (ViltProcessor): Processor used to encode the image and text.
        batch_size (int): Batch size.
        device (torch.device): Device to run the model on.
        
    Returns:
        list: List of predictions.
    """
    
    # Load the processor and model
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
    model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
    model.to(device)

    # Load the classifier
    num_classes = 2
    classifier = nn.Linear(model.config.hidden_size, num_classes).to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load('model_final.pt', map_location=device))
    model.eval()
    
    def ex_collate_fn(batch):
        return collate_fn(batch, processor)

    dataset = MultimodalDataset(csv_file=dataset_path, processor=processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=ex_collate_fn)
    
    num_classes = 2
    classifier = nn.Linear(model.config.hidden_size, num_classes).to(device)

    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    
    all_preds = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='Inference'):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                pixel_mask=batch["pixel_mask"]
            ) 
            
            pooler_output = outputs.pooler_output
            logits = classifier(pooler_output)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            
    return all_preds
