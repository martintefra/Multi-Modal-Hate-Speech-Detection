import tqdm
import torch
import torch.nn as nn
from sklearn.utils import resample
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import ViltProcessor, ViltModel, get_scheduler, AdamW

from models import MultimodalDataset
from utils import collate_fn


# Load the pre-trained ViLT model and processor
model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Add a classifier on top of the model for binary classification
num_classes = 2
classifier = nn.Linear(model.config.hidden_size, num_classes).to(device)

# Load dataset and take a subset
dataset = MultimodalDataset(csv_file='data/MMHS150K_GT.csv', processor=processor)
dataset = resample(dataset, n_samples=20000, random_state=42)

# Split dataset into training, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create dataloaders for training, validation, and test sets
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,  collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  collate_fn=collate_fn)

# Set up the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()
num_epochs = 4
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
    all_preds = []
    all_labels = []
    for batch in tqdm.tqdm(train_dataloader, desc='Training'):
        batch = {k:v.to(device) for k,v in batch.items()}

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
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

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        total_correct += torch.sum(preds == batch["labels"]).item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())
    
    train_loss = total_loss / len(train_dataloader)
    train_acc = total_correct / len(train_dataset)
    train_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Training Loss: {train_loss:.4f}')
    print(f'Training Accuracy: {train_acc:.4f}')
    print(f'Training F1 Score: {train_f1:.4f}')
        
        
    model.eval()
    total_loss = 0
    total_correct = 0
    all_preds = []
    all_labels = []
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
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            

    val_loss = total_loss / len(val_dataloader)
    val_acc = total_correct / len(val_dataset)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_acc:.4f}')
    print(f'Validation F1 Score: {val_f1:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'model.pt')
        
# Testing the model
model.load_state_dict(torch.load('model.pt'))
model.eval()
total_correct = 0
all_preds = []
all_labels = []

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
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())
        
test_acc = total_correct / len(test_dataset)
test_f1 = f1_score(all_labels, all_preds, average='weighted')
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test F1 Score: {test_f1:.4f}')

# save the model
torch.save(model.state_dict(), 'model_final.pt')