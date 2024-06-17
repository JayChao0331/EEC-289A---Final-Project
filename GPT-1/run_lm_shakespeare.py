from collections import Counter
from nltk.corpus import shakespeare

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.optim as optim
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTConfig

from read_data.read_shakespeare import extract_sentences_from_work, clean_text, validate_text, train_test_split


# Hyper-parameters
N = 2
pretrained = True
num_epochs = 10


def extract_unique_n_minus_1_grams(corpus, n):
    n_minus_1_grams_freq = Counter()
    total_n_minus_1_grams = 0
    for sentence in corpus:
        sentence = ''.join(sentence)
        for i in range(len(sentence) - n + 2):
            context = sentence[i:i + n - 1]
            n_minus_1_grams_freq[context] += 1
            total_n_minus_1_grams += 1
    
    # Normalize the frequencies
    n_minus_1_grams_freq_normalized = {context: freq / total_n_minus_1_grams for context, freq in n_minus_1_grams_freq.items()}
    
    return n_minus_1_grams_freq_normalized


class BigramDataset(Dataset):
    def __init__(self, corpus, train=False, valid=False, test=True):
        self.corpus = corpus
        self.vocab = list('abcdefghijklmnopqrstuvwxyz ')
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

        # Split the data into training and validation sets
        data = self.create_bigram_pairs()

        if train:
            train_size = int(len(data) * 0.9)
            self.data = data[:train_size]
        elif valid:
            train_size = int(len(data) * 0.9)
            self.data = data[train_size:]
        elif test:
            self.data = data

    def create_bigram_pairs(self):
        data = []
        for sentence in self.corpus:
            for i in range(len(sentence) - 1):
                input_char = sentence[i]
                output_char = sentence[i + 1]
                if input_char in self.char_to_idx and output_char in self.char_to_idx:
                    data.append((self.char_to_idx[input_char], self.char_to_idx[output_char]))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0], dtype=torch.long), torch.tensor(self.data[idx][1], dtype=torch.long)


class TrigramDataset(Dataset):
    def __init__(self, corpus, train=False, valid=False, test=True):
        self.corpus = corpus
        self.vocab = list('abcdefghijklmnopqrstuvwxyz ')
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

        # Split the data into training and validation sets
        data = self.create_trigram_pairs()

        if train:
            train_size = int(len(data) * 0.9)
            self.data = data[:train_size]
        elif valid:
            train_size = int(len(data) * 0.9)
            self.data = data[train_size:]
        elif test:
            self.data = data

    def create_trigram_pairs(self):
        data = []
        for sentence in self.corpus:
            for i in range(len(sentence) - 2):
                input_chars = (sentence[i], sentence[i + 1])
                output_char = sentence[i + 2]
                if input_chars[0] in self.char_to_idx and input_chars[1] in self.char_to_idx and output_char in self.char_to_idx:
                    data.append((self.char_to_idx[input_chars[0]], self.char_to_idx[input_chars[1]], self.char_to_idx[output_char]))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx][0], dtype=torch.long), 
                torch.tensor(self.data[idx][1], dtype=torch.long), 
                torch.tensor(self.data[idx][2], dtype=torch.long))
    

def train(model, criterion, optimizer, train_loader, valid_loader, device):
    # Training loop
    min_loss = 10000
    train_loss = 0
    valid_loss = 0
    for epoch in range(num_epochs):

        if N == 2:
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                inputs = inputs.unsqueeze(1).to(device)  # Model expects sequence, so adding sequence length dimension
                targets = targets.to(device)
                outputs = model(inputs).logits
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs = inputs.unsqueeze(1).to(device)  # Model expects sequence, so adding sequence length dimension
                    targets = targets.to(device)
                    outputs = model(inputs).logits
                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets)
                    valid_loss += loss.item()

            print(f"Epoch {epoch + 1}, Training Loss: {train_loss / len(train_loader)}, Validation Loss: {valid_loss / len(valid_loader)}")
        elif N == 3:
            model.train()
            train_loss = 0
            for input1, input2, targets in train_loader:
                optimizer.zero_grad()
                inputs = torch.stack([input1, input2], dim=1)
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs).logits
                outputs = outputs[:, -1, :]
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for input1, input2, targets in valid_loader:
                    inputs = torch.stack([input1, input2], dim=1)
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs).logits
                    outputs = outputs[:, -1, :]
                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                    valid_loss += loss.item()

            print(f"Epoch {epoch + 1}, Training Loss: {train_loss / len(train_loader)}, Validation Loss: {valid_loss / len(valid_loader)}")

        if valid_loss <= min_loss:
            # Save the trained model
            min_loss = valid_loss
            if pretrained:
                torch.save(model.state_dict(), '{}gram_shakespeare_pretrained_gpt_model.pth'.format(N))
            else:
                torch.save(model.state_dict(), '{}gram_shakespeare_without_pretrained_gpt_model.pth'.format(N))


def test(model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        if N == 2:
            for inputs, targets in test_loader:
                inputs = inputs.unsqueeze(1).to(device)  # Add dummy dimension for sequence length
                targets = targets.to(device)
                outputs = model(inputs).logits
                _, predicted = torch.max(outputs, 2)
                total += targets.size(0)
                correct += (predicted.squeeze() == targets).sum().item()
        elif N == 3:
            for input1, input2, targets in test_loader:
                inputs = torch.stack([input1, input2], dim=1)
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs).logits
                outputs = outputs[:, -1, :]
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted.squeeze() == targets).sum().item()

    print(f"Test Accuracy: {correct / total * 100:.2f}%")


def calculate_prediction_entropy(model, n_minus_1_grams_freq, dataset, n, device):
    entropy_dict = {}

    for context, freq in n_minus_1_grams_freq.items():
        context_idx = torch.tensor([dataset.char_to_idx[char] for char in context], dtype=torch.long)
        context_idx = context_idx.unsqueeze(0)  # Add batch dimension
        context_idx = context_idx.to(device)

        with torch.no_grad():
            outputs = model(context_idx).logits
            probabilities = F.softmax(outputs, dim=-1).squeeze(0)

        # We use the probabilities of the next character directly
        next_char_probs = probabilities[-1]
        
        entropy = -torch.sum(next_char_probs * torch.log(next_char_probs + 1e-9)).item()
        entropy_dict[context] = entropy

    return entropy_dict


def calculate_weighted_sum(entropies, n_minus_1_grams_freq_normalized):
    weighted_sum = 0
    for context, entropy in entropies.items():
        weighted_sum += entropy * n_minus_1_grams_freq_normalized.get(context, 0)
    return weighted_sum





if __name__ == '__main__':
    # Read Data
    works = shakespeare.fileids()

    all_works_sentences = []
    for work in works:
        sentences = extract_sentences_from_work(work)
        cleaned_sentences = [list(clean_text(sentence)) for sentence in sentences]
        all_works_sentences.append(cleaned_sentences)

    if validate_text(all_works_sentences):
        print('Dictionary size is correct!')
    else:
        print('Dictionary size is wrong!')

    train_set, test_set = train_test_split(all_works_sentences)

    # Create datasets and dataloaders
    train_dataset = None
    valid_dataset = None
    test_dataset = None
    test_n_minus_1_grams_freq = None
    if N == 2:
        train_dataset = BigramDataset(train_set, train=True, valid=False, test=False)
        valid_dataset = BigramDataset(train_set, train=False, valid=True, test=False)
        test_dataset = BigramDataset(test_set, train=False, valid=False, test=True)
        test_n_minus_1_grams_freq = extract_unique_n_minus_1_grams(test_set, n=N)
    elif N == 3:
        train_dataset = TrigramDataset(train_set, train=True, valid=False, test=False)
        valid_dataset = TrigramDataset(test_set, train=False, valid=True, test=False)
        test_dataset = TrigramDataset(test_set, train=False, valid=False, test=True)
        test_n_minus_1_grams_freq = extract_unique_n_minus_1_grams(test_set, n=N)
    
    print("\ntraining set: {}".format(len(train_dataset)))
    print("validation set: {}".format(len(valid_dataset)))
    print("testing set: {}".format(len(test_dataset)))
    print("test_n_minus_1_grams_freq: {}\n".format(len(test_n_minus_1_grams_freq)))

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False)

    # Define OpenAI GPT configuration and model
    config = None
    model = None
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if pretrained:
        model_name = 'openai-gpt'
        model = OpenAIGPTLMHeadModel.from_pretrained(model_name)
        model.resize_token_embeddings(len(train_dataset.char_to_idx))
    else:
        config = OpenAIGPTConfig(vocab_size=len(train_dataset.char_to_idx), n_positions=512, n_embd=128, n_layer=6, n_head=8)
        model = OpenAIGPTLMHeadModel(config)
    
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Start training
    print("Start training...\n")
    train(model, criterion, optimizer, train_loader, valid_loader, device)

    # Start testing
    print("Start testing...\n")
    if pretrained:
        model_name = 'openai-gpt'
        model = OpenAIGPTLMHeadModel.from_pretrained(model_name)
        model.resize_token_embeddings(len(train_dataset.char_to_idx))
    else:
        config = OpenAIGPTConfig(vocab_size=len(train_dataset.char_to_idx), n_positions=512, n_embd=128, n_layer=6, n_head=8)
        model = OpenAIGPTLMHeadModel(config)
    
    model_path = None
    if pretrained:
        model_path = '{}gram_shakespeare_pretrained_gpt_model.pth'.format(N)
    else:
        model_path = '{}gram_shakespeare_without_pretrained_gpt_model.pth'.format(N)

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    test(model, test_loader, device)
    entropy_dict = calculate_prediction_entropy(model, test_n_minus_1_grams_freq, test_dataset, n=N, device=device)
    entropy = calculate_weighted_sum(entropy_dict, test_n_minus_1_grams_freq)

    print("Estimated entropy: {}".format(entropy))
