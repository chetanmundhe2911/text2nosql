import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# Custom Dataset
class MultiTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = " [SEP] ".join(self.texts[index])  # Concatenate texts
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Load data (you can replace this with your MongoDB loading code)
data = {
    'questions': [
        ["What is the capital of France?", "Tell me about Paris."],
        ["How to cook pasta?", "What are some pasta recipes?"],
        ["Explain quantum physics.", "What is entanglement?"],
        ["What is the best way to learn Python?", "Can you suggest some resources?"]
    ],
    'labels': [
        ['geography'],
        ['cooking'],
        ['science', 'physics'],
        ['programming', 'education']
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# MultiLabel Binarizer
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['labels'])

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(df['questions'], y, test_size=0.2, random_state=42)

# Parameters
MAX_LEN = 128
BATCH_SIZE = 4
EPOCHS = 3
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Datasets and DataLoaders
train_dataset = MultiTextDataset(X_train.tolist(), y_train.tolist(), tokenizer, MAX_LEN)
val_dataset = MultiTextDataset(X_val.tolist(), y_val.tolist(), tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=y.shape[1])
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
        attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
        labels = batch['labels'].to('cuda' if torch.cuda.is_available() else 'cpu')

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"Epoch {epoch + 1}/{EPOCHS} completed.")

# Validation Loop
model.eval()
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
        attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
        labels = batch['labels'].to('cuda' if torch.cuda.is_available() else 'cpu')

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = (outputs.logits.sigmoid() > 0.5).int()
        
        correct_predictions += (preds == labels).sum().item()
        total_predictions += labels.numel()

    print(f"Validation Accuracy: {correct_predictions / total_predictions:.4f}")
