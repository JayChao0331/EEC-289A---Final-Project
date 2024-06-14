import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.corpus import shakespeare, brown
from nltk.util import ngrams
from collections import Counter

def preprocess_text(corpus):
    return ' '.join(corpus)

def get_weighted_n_minus_1_grams(corpus, n):
    n_grams = list(ngrams(corpus, n))
    n_minus_1_grams = [gram[:-1] for gram in n_grams]
    n_minus_1_gram_frequencies = Counter(n_minus_1_grams)
    total_count = sum(n_minus_1_gram_frequencies.values())
    weighted_grams = {gram: count / total_count for gram, count in n_minus_1_gram_frequencies.items()}
    return weighted_grams

model_load_path = '/Users/howard/Documents/PyCharmProjects/EEC289A/final_project/distilgpt2_fine_tuning'
tokenizer = AutoTokenizer.from_pretrained(model_load_path)
model = AutoModelForCausalLM.from_pretrained(model_load_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

character_vocabulary = list("abcdefghijklmnopqrstuvwxyz ")
char_to_token_id = {char: tokenizer.encode(char, add_special_tokens=False)[0] for char in character_vocabulary}

def calculate_average_entropy(corpus, n):
    data = preprocess_text(corpus)
    weighted_n_minus_1_grams = get_weighted_n_minus_1_grams(data, n)
    n_minus_1_gram_keys = weighted_n_minus_1_grams.keys()

    n_minus_1_grams_list = []
    for gram in n_minus_1_gram_keys:
        n_minus_1_grams_list.append([char for char in gram])

    entropies = []

    for gram in n_minus_1_grams_list:
        input_text = ''.join(gram)
        input_ids = torch.tensor([tokenizer.encode(input_text, add_special_tokens=False)], device=device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        next_token_logits = logits[:, -1, :].squeeze()
        char_logits = [next_token_logits[char_to_token_id[char]].item() for char in character_vocabulary]
        char_logits_tensor = torch.tensor(char_logits)
        char_probs = torch.softmax(char_logits_tensor, dim=-1)

        char_probs_tensor = torch.tensor(char_probs)
        entropy = -torch.sum(char_probs_tensor * torch.log(char_probs_tensor + 1e-9))

        entropies.append(entropy.item())

    average_entropy = sum(weight * entropy for weight, entropy in zip(weighted_n_minus_1_grams.values(), entropies))

    return average_entropy

shakespeare_corpus = shakespeare.words('hamlet.xml')
average_entropy_shakespeare = calculate_average_entropy(shakespeare_corpus, n=3)
print(f"Average Entropy for Shakespeare: {average_entropy_shakespeare}")

brown_corpus = brown.words(categories='news')
average_entropy_brown = calculate_average_entropy(brown_corpus, n=3)
print(f"Average Entropy for Brown: {average_entropy_brown}")
