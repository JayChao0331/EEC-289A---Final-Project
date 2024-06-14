# Load model directly
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


from nltk.corpus import brown
from read_brown import preprocess_word, preprocess_corpus, tokenizer, train_test_split
from sample_code_shakespeare import get_train_test_data, extract_unique_n_minus_1_grams
from my_read_brown import n_minus_1_gram
import json

#def extract_unique_n_minus_1_grams(corpus, n):
#    n_minus_1_grams = set()
##    for sentence in corpus:
#        sentence = ''.join(sentence)
#        for i in range(len(sentence) - n + 2):
#            context = sentence[i:i + n - 1]
#            n_minus_1_grams.add(context)
#    return n_minus_1_grams

#def get_train_test_data():
#    # Read Data
#    corpus = brown.sents()
#    corpus_processed = preprocess_corpus(corpus)
#    corpus_tokenized = tokenizer(corpus_processed)
#    train_set, test_set = train_test_split(corpus_tokenized, ratio=0.7)
#
#    return train_set, test_set

n_m_gram = extract_unique_n_minus_1_grams(brown.sents(), 2)
sent, tr, te = get_train_test_data()

n = 3
weighted_grams = extract_unique_n_minus_1_grams(te,n)

grim = weighted_grams.keys()

n_minus_1 = []

if n >= 3:
    for grom in grim:
        n_minus_1.append([x for x in grom])
else:
    n_minus_1 = list(grim)


#print(n_m_gram)



tokenizer = AutoTokenizer.from_pretrained("astronomer/Llama-3-8B-GPTQ-4-Bit")
model = AutoModelForCausalLM.from_pretrained("astronomer/Llama-3-8B-GPTQ-4-Bit", device_map="cuda:0")

#print(tokenizer.get_vocab)

model.eval()

# Ensure the model uses GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Custom character vocabulary
char_vocab = list("abcdefghijklmnopqrstuvwxyz ")
char_to_token_id = {ch: tokenizer.encode(ch, add_special_tokens=False)[0] for ch in char_vocab}

#n_minus_1 = n_minus_1_gram(2)
entropies = []

for grum in range(1):#n_minus_1:
    # Input text for trigram prediction
    input_text = ['t', 'h']#grum  # Change as needed
    input_ids = torch.tensor([tokenizer.encode(input_text, add_special_tokens=False)], device=device)
    #print(input_ids)

    # Get logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Filter logits before softmax
    next_token_logits = logits[:, -1, :].squeeze()
    char_logits = [next_token_logits[char_to_token_id[ch]].item() for ch in char_vocab]
    char_logits_tensor = torch.tensor(char_logits)
    char_probs = torch.softmax(char_logits_tensor, dim=-1)

    # Compute probabilities over the entire vocabulary
    probs = torch.softmax(logits, dim=-1)

    # Get probabilities for the next character
    next_token_probs = probs[:, -1, :].squeeze()

    # Extract relevant probabilities for characters in custom vocabulary
    #char_probs = [next_token_probs[char_to_token_id[ch]].item() for ch in char_vocab]

    # Calculate entropy
    char_probs_tensor = torch.tensor(char_probs)
    entropy = -torch.sum(char_probs_tensor * torch.log(char_probs_tensor + 1e-9))  # Adding a small value to avoid log(0)

    # Print entropy
    #print(f"Entropy: {entropy.item()}")
    entropies.append(entropy.item())

    # Print sorted probabilities
    char_probs_dict = {ch: next_token_probs[char_to_token_id[ch]].item() for ch in char_vocab}
    sorted_char_probs = sorted(char_probs_dict.items(), key=lambda item: item[1], reverse=True)
    #for char, prob in sorted_char_probs:
        #print(f"Character: {char}, Probability: {prob}")

print(char_logits)
print(type(entropies[0]))

tensor_entropies = torch.FloatTensor(entropies)

#avg_entropy_shake = torch.mean(tensor_entropies)
avg_entropy_shake = 0
i = 0
for weight in weighted_grams.values():
    avg_entropy_shake += weight*entropies[i]
    i+=1

print(avg_entropy_shake)

