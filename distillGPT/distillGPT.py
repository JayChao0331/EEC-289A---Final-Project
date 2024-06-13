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
<<<<<<< HEAD
    def __init__(self, text, tokenizer, block_size=128):
        self.examples = []
        processed_text = preprocess_text(text)
        ngrams = [processed_text[i:i + block_size] for i in range(0, len(processed_text) - block_size + 1, block_size)]
        for ngram in ngrams:
            self.examples.append(tokenizer.encode(ngram, add_special_tokens=True))
=======
    def __init__(self, text, tokenizer, n_gram=3):
        self.examples = []
        processed_text = preprocess_text(text)
        self.tokenizer = tokenizer
        self.n_gram = n_gram

        tokens = tokenizer.tokenize(processed_text)
        for i in range(len(tokens) - n_gram + 1):
            input_tokens = tokens[i:i + (n_gram - 1)]
            target_token = tokens[i + (n_gram - 1)]
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            target_id = tokenizer.convert_tokens_to_ids([target_token])[0]
            self.examples.append((input_ids, target_id))
>>>>>>> 73f3ea2e7b207c66643ee9352845e7cfdc7db648

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
<<<<<<< HEAD
        return torch.tensor(self.examples[item], dtype=torch.long)

def train(model, tokenizer, dataset, epochs=5, batch_size=32, lr=5e-5):
    train_size = int(0.7 * len(dataset))  # 70% for training sets
    test_size = len(dataset) - train_size  # 30% for testing sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
=======
        input_ids, target_id = self.examples[item]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_id, dtype=torch.long)

def train(model, tokenizer, dataset, epochs=5, batch_size=32, lr=5e-5):
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
>>>>>>> 73f3ea2e7b207c66643ee9352845e7cfdc7db648

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_loader) * epochs)

<<<<<<< HEAD
    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            inputs = batch.to(model.device)
=======
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress_bar:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
>>>>>>> 73f3ea2e7b207c66643ee9352845e7cfdc7db648
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
<<<<<<< HEAD
            inputs = batch.to(model.device)
=======
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
>>>>>>> 73f3ea2e7b207c66643ee9352845e7cfdc7db648
            outputs = model(inputs, labels=inputs)
            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Average test loss: {avg_loss}")

<<<<<<< HEAD

# url = "https://www.gutenberg.org/files/100/100-0.txt"             #Shakespear Corpus
# text = load_text_from_url(url)
text = ' '.join(brown.words())                                      #Brown Corpus
dataset = TextDataset(text, tokenizer)

train(model, tokenizer, dataset, epochs=10, batch_size=32, lr=5e-5)
=======
    return test_dataset

def print_probabilities(model, tokenizer, dataset, n_gram=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        for input_ids, target_id in dataset:
            input_tensor = input_ids.unsqueeze(0).to(device)

            outputs = model(input_tensor)
            logits = outputs.logits[0, -1, :]
            probabilities = torch.softmax(logits, dim=-1)

            input_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
            print(f"Context: {''.join(input_tokens)}")
            for char, prob in zip(tokenizer.convert_ids_to_tokens(range(len(tokenizer))), probabilities):
                if char in 'abcdefghijklmnopqrstuvwxyz ':
                    print(f"{char}: {prob:.4f}")
            print("\n")

text = ' '.join(brown.words())

dataset = TextDataset(text, tokenizer, n_gram=3)

test_dataset = train(model, tokenizer, dataset, epochs=10, batch_size=32, lr=5e-5)

print_probabilities(model, tokenizer, test_dataset, n_gram=3)
>>>>>>> 73f3ea2e7b207c66643ee9352845e7cfdc7db648
