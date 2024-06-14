import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup
from nltk.corpus import shakespeare, brown
from nltk.util import ngrams
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import os


def preprocess_text(corpus):
    return ' '.join(corpus)


class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, block_size):
        self.examples = []
        for text in texts:
            tokenized_text = tokenizer.encode(text, add_special_tokens=True)
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                self.examples.append(tokenized_text[i:i + block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
BLOCK_SIZE = 128

config = GPT2Config()
model = GPT2LMHeadModel(config)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

shakespeare_corpus = shakespeare.words('hamlet.xml')
brown_corpus = brown.words(categories='news')

train_texts = [preprocess_text(shakespeare_corpus), preprocess_text(brown_corpus)]
train_dataset = TextDataset(tokenizer, train_texts, BLOCK_SIZE)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

model.train()
for epoch in range(EPOCHS):
    for batch in train_dataloader:
        batch = batch.to(device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item()}")

model_save_path = '/Users/howard/Documents/PyCharmProjects/EEC289A/final_project/distilgpt2_train from scratch'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")
