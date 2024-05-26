import torch

def collate_fn(batch, processor, pad_sequence):
    """
    Collate function to pad batch data.

    Args:
        batch (list): List of dictionaries where each dictionary represents a sample.
        processor (ViltProcessor): Processor used to encode the image and text.
        pad_sequence (torch.nn.utils.rnn.pad_sequence): Function to pad the input_ids, attention_mask, and token_type_ids.

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
