import re
import nltk
from nltk.corpus import shakespeare

nltk.download('shakespeare')


def extract_sentences_from_work(work):
    work_text = shakespeare.raw(work)
    sentences = []

    # Use regular expressions to find <SPEAKER> and <LINE> tags and extract the text
    speaker_pattern = re.compile(r'<SPEAKER>(.*?)</SPEAKER>', re.DOTALL)
    line_pattern = re.compile(r'<LINE>(.*?)</LINE>', re.DOTALL)

    speakers = speaker_pattern.findall(work_text)
    lines = line_pattern.findall(work_text)

    # Combine speaker and line text into sentences
    for speaker, line in zip(speakers, lines):
        # Remove any leading or trailing whitespace from the speaker and line
        speaker = speaker.strip()
        line = line.strip()
        # Combine speaker and line into a sentence
        sentence = f"{speaker}: {line}"
        sentences.append(sentence)

    return sentences


def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove characters that are not a-z or space
    cleaned_text = re.sub(r'[^a-z\s]', '', text)
    return cleaned_text


def validate_text(all_works_sentences):
    valid = True
    for work_sentences in all_works_sentences:
        for sentence in work_sentences:
            if not all(char in 'abcdefghijklmnopqrstuvwxyz ' for char in sentence):
                valid = False
                return valid
    return valid


def train_test_split(corpus):
    corpus_len = len(corpus)
    train_set_tmp = corpus[:corpus_len-1]
    train_set = []
    for work_lst in train_set_tmp:
        for paragraph in work_lst:
            train_set.append(paragraph)

    test_set_tmp = corpus[corpus_len-1:]
    test_set = []
    for work_lst in test_set_tmp:
        for paragraph in work_lst:
            test_set.append(paragraph)

    return train_set, test_set
