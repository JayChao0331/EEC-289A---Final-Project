import requests
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, random_split
import re
from tqdm import tqdm
import nltk
from nltk.corpus import brown

nltk.download('brown')

def load_text_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def collate_fn(batch):
    input_ids = [item for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    return input_ids

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size=128):
        self.examples = []
        processed_text = preprocess_text(text)
        ngrams = [processed_text[i:i + block_size] for i in range(0, len(processed_text) - block_size + 1, block_size)]
        for ngram in ngrams:
            self.examples.append(tokenizer.encode(ngram, add_special_tokens=True))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

def train(model, tokenizer, dataset, epochs=5, batch_size=32, lr=5e-5):
    train_size = int(0.7 * len(dataset))  # 70% for training sets
    test_size = len(dataset) - train_size  # 30% for testing sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_loader) * epochs)

    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            inputs = batch.to(model.device)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.set_postfix(loss=loss.item())

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch.to(model.device)
            outputs = model(inputs, labels=inputs)
            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Average test loss: {avg_loss}")


# url = "https://www.gutenberg.org/files/100/100-0.txt"             #Shakespear Corpus
# text = load_text_from_url(url)
text = ' '.join(brown.words())                                      #Brown Corpus
dataset = TextDataset(text, tokenizer)

train(model, tokenizer, dataset, epochs=10, batch_size=32, lr=5e-5)
