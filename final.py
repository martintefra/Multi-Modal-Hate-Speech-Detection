import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViltProcessor, ViltModel, get_scheduler, AdamW
from torch.nn.utils.rnn import pad_sequence

from models import MultimodalDataset
from sklearn.utils import resample


# Load the model
model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")

# else "mps" if torch.backends.mps.is_available() 
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Classifier added to the model
num_classes = 2
classifier = nn.Linear(model.config.hidden_size, num_classes).to(device)

def collate_fn(batch):
    # Extract features and labels from batch
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

dataset = MultimodalDataset(csv_file='data/MMHS150K_GT.csv', processor=processor)

# take subset of dataset
dataset = resample(dataset, n_samples=10000, random_state=42)

train_size = int(0.8 * len(dataset))
val_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# create dataloaders
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,  collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  collate_fn=collate_fn)


# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Learning rate scheduler
criterion = nn.CrossEntropyLoss()
num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Training loop
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_correct = 0
    for batch in tqdm.tqdm(train_dataloader, desc='Training'):
        batch = {k:v.to(device) for k,v in batch.items()}

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(
            input_ids=batch["input_ids"],
            pixel_values=batch["pixel_values"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            pixel_mask=batch["pixel_mask"]
        ) 
        
        pooler_output = outputs.pooler_output
        
        logits = classifier(pooler_output)
        
        loss = criterion(logits, batch["labels"].long())
        print(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        total_correct += torch.sum(preds == batch["labels"]).item()
    
    train_loss = total_loss / len(train_dataloader)
    train_acc = total_correct / len(train_dataset)
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Training Loss: {train_loss:.4f}')
    print(f'Training Accuracy: {train_acc:.4f}')
        
        
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(val_dataloader, desc='Evaluating'):
            batch = {k:v.to(device) for k,v in batch.items()}
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                pixel_mask=batch["pixel_mask"]
            ) 
            
            pooler_output = outputs.pooler_output
            
            logits = classifier(pooler_output)
            
            loss = criterion(logits, batch["labels"].long())
        
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += torch.sum(preds == batch["labels"]).item()
            

    val_loss = total_loss / len(val_dataloader)
    val_acc = total_correct / len(val_dataset)
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_acc:.4f}')
    print()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'model.pt')
        
# testing
model.load_state_dict(torch.load('model.pt'))
model.eval()
total_correct = 0
with torch.no_grad():
    for batch in tqdm.tqdm(test_dataloader, desc='Testing'):
        batch = {k:v.to(device) for k,v in batch.items()}
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
        total_correct += torch.sum(preds == batch["labels"]).item()
        
test_acc = total_correct / len(test_dataset)
print(f'Test Accuracy: {test_acc:.4f}')