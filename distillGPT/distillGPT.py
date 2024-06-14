import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re
import nltk
from nltk.corpus import shakespeare
from collections import Counter
from torch.utils.data import Dataset, random_split

nltk.download('shakespeare')

model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

class TextDataset(Dataset):
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

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        input_ids, target_id = self.examples[item]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_id, dtype=torch.long)

def print_probabilities_and_entropy(model, tokenizer, dataset, n_gram=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    bigram_counts = Counter()
    for input_ids, target_id in dataset:
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
        bigram = ''.join(input_tokens)
        bigram_counts[bigram] += 1

    total_bigrams = sum(bigram_counts.values())
    weighted_entropy = 0

    with torch.no_grad():
        for input_ids, target_id in dataset:
            input_tensor = input_ids.unsqueeze(0).to(device)
            outputs = model(input_tensor)
            logits = outputs.logits[0, -1, :]

            print(f"Logits range: min={logits.min().item()}, max={logits.max().item()}")

            scaled_logits = (logits - logits.mean()) / logits.std()

            probabilities = torch.softmax(scaled_logits, dim=-1)

            input_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
            context = ''.join(input_tokens).replace('Ġ', ' ') 

            if len(context) > 2:
                context = context[-2:]

            char_probs = {char: prob.item() for char, prob in zip(tokenizer.convert_ids_to_tokens(range(len(tokenizer))), probabilities) if len(char) == 1 and char in 'abcdefghijklmnopqrstuvwxyz '}
            total_prob = sum(char_probs.values())

            for char in char_probs:
                char_probs[char] /= total_prob

            entropy = -sum(prob * torch.log2(torch.tensor(prob + 1e-12)).item() for prob in char_probs.values())
            weight = bigram_counts[context] / total_bigrams
            weighted_entropy += entropy * weight

            print(f"Context: {context}")
            for char, prob in char_probs.items():
                print(f"{char}: {prob:.4f}") 
            print(f"Total Probability: {sum(char_probs.values()):.4f}")
            print(f"Entropy: {entropy:.4f}, Weight: {weight:.4f}, Weighted Entropy: {entropy * weight:.4f}\n")

    print(f"Total Weighted Entropy: {weighted_entropy:.4f}")

shakespeare_files = shakespeare.fileids()
text = ' '.join([word for fileid in shakespeare_files for word in shakespeare.words(fileid)])

dataset = TextDataset(text, tokenizer, n_gram=3)

train_size = int(0.7 * len(dataset)) 
test_size = len(dataset) - train_size  
_, test_dataset = random_split(dataset, [train_size, test_size])

print_probabilities_and_entropy(model, tokenizer, test_dataset, n_gram=3)
