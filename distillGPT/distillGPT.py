import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import re
import requests
from collections import defaultdict

model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def predict_next_char_and_entropy(context, model, tokenizer):
    input_ids = tokenizer.encode(context, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        probabilities = torch.softmax(next_token_logits, dim=-1).squeeze()

        # 過濾掉非小寫字母和空白字符的概率
        valid_char_indices = [i for i in range(len(probabilities)) if re.match(r'^[a-z ]$', tokenizer.decode([i]))]
        valid_probabilities = probabilities[valid_char_indices]
        valid_tokens = [tokenizer.decode([i]) for i in valid_char_indices]

        # 正規化概率分佈
        total_probability = valid_probabilities.sum()
        normalized_probabilities = valid_probabilities / total_probability

        entropy = -sum(p * torch.log(p) for p in normalized_probabilities if p > 0)

        predicted_token_index = normalized_probabilities.argmax()
        next_char = valid_tokens[predicted_token_index]

    return next_char, dict(zip(valid_tokens, normalized_probabilities)), entropy.item()

def load_text_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

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


def train(model, tokenizer, dataset, epochs=1, batch_size=4, lr=5e-5):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_loader) * epochs)

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            inputs = batch.to(model.device)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

url = "https://www.gutenberg.org/files/100/100-0.txt"
text = load_text_from_url(url)
dataset = TextDataset(text, tokenizer)
train(model, tokenizer, dataset, epochs=1, batch_size=32, lr=5e-5)  # 修改epochs參數
processed_text = preprocess_text(text)
N = 3 
ngrams = [processed_text[i:i + N - 1] for i in range(len(processed_text) - N + 1)]

for context in ngrams:
    next_char, probabilities, entropy = predict_next_char_and_entropy(context, model, tokenizer)
    print(f"Context: {context}, Next Char: {next_char}, Entropy: {entropy}")
    # print(f"Probabilities: {probabilities}")
    # sorted_probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
    # print(f"Sorted Probabilities: {sorted_probabilities}")
    # print(f"Total Probability: {sum(probabilities.values())}")
