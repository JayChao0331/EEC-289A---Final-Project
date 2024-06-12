from transformers import PreTrainedTokenizer, TFOpenAIGPTLMHeadModel
from transformers import OpenAIGPTLMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import OpenAIGPTConfig
import torch
import tensorflow as tf
from nltk.corpus import brown
import re
import nltk
from collections import Counter
from datasets import Dataset


nltk.download('brown')


class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab):
        self.vocab = vocab
        super().__init__(vocab=vocab, pad_token="<pad>", unk_token="<unk>")

        self.ids_to_tokens = {v: k for k, v in vocab.items()}
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.pad_token_id = vocab[self.pad_token]

    def _tokenize(self, text):
        return list(text)

    def convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get("<unk>"))

    def convert_ids_to_tokens(self, index):
        return self.ids_to_tokens.get(index, "<unk>")

    def get_vocab(self):
        return self.vocab

    def encode(self, text, max_length=None, padding=False, truncation=False):
        tokens = self._tokenize(text)
        token_ids = [self.convert_token_to_id(token) for token in tokens]
        if truncation and max_length:
            token_ids = token_ids[:max_length]
        if padding and max_length:
            token_ids += [self.convert_token_to_id(self.pad_token)] * (max_length - len(token_ids))
        #attention_mask = [1] * len(token_ids)  # Add attention mask
        return token_ids

    def decode(self, token_ids):
        tokens = [self.convert_ids_to_tokens(token_id) for token_id in token_ids]
        return ''.join(tokens)



print("Preprocessing corpus")
# Preprocess and tokenize the corpus
corpus = list(brown.sents())
corpus_processed = preprocess_corpus(corpus)
corpus_tokenized = tokenizer(corpus_processed)

print("Tokenized corpus")
# Create vocabulary
flat_corpus = [item for sublist in corpus_tokenized for item in sublist]
vocab_counter = Counter(flat_corpus)
vocab = {word: idx for idx, (word, _) in enumerate(vocab_counter.items(), start=1)}
vocab["<pad>"] = 0  # Add padding token

print("Generating vocab")

# Prepare data for Dataset
texts = [''.join(sentence) for sentence in corpus_tokenized]
train_data = {'text': texts}
train_dataset = Dataset.from_dict(train_data)

print("Constructing Training Dataset")
# Initialize the custom tokenizer
custom_tokenizer = CustomTokenizer(vocab)

print("Custom Tokenizer Made")


# Tokenize the data
def tokenize_function(examples):
    encodings = [custom_tokenizer.encode(text, max_length=512, padding=True, truncation=True) for text in
                 examples['text']]
    max_len = max(len(encoding) for encoding in encodings)
    padded_encodings = [encoding + [custom_tokenizer.pad_token_id] * (max_len - len(encoding)) for encoding in
                        encodings]
    return {'input_ids': padded_encodings}


train_dataset = train_dataset.map(tokenize_function, batched=True)

print("Map trainted to tokenzied")

# Ensure TensorFlow is set to use the GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("FOUND GPU!")
else:
    print("No GPU found. Using CPU.")

# Load the GPT-1 model
model = TFOpenAIGPTLMHeadModel.from_pretrained("openai-community/openai-gpt")
print("Model Loaded")



#configuration = OpenAIGPTConfig()

#model = TFOpenAIGPTLMHeadModel(configuration)

#tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
#model = TFOpenAIGPTLMHeadModel.from_pretrained("openai-community/openai-gpt")

iconfig = OpenAIGPTConfig()
model = OpenAIGPTLMHeadModel(config)
model.resize_token_embeddings(len(vocab))

tokenizer = CustomTokenizer(vocab)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model Loaded")



data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,  # Custom tokenizer does not need to be passed here
    mlm=False, 
)



training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

trainer.train()

