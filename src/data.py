"""SST-2 sentences → CLS-prefixed padded id tensors."""

import re
import unicodedata

import torch
from torch.utils.data import DataLoader, Dataset

PAD_token = 0
CLS_token = 1
UNK_token = 2

MAX_LENGTH = 64
BATCH_SIZE = 32


def unicodeToAscii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


class Input:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<pad>": PAD_token, "<cls>": CLS_token, "<unk>": UNK_token}
        self.word2count = {"<pad>": 0, "<cls>": 0, "<unk>": 0}
        self.index2word = {PAD_token: "<pad>", CLS_token: "<cls>", UNK_token: "<unk>"}
        self.n_words = 3

    def addSentence(self, sentence):
        for word in sentence.split():
            if word.strip():
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def indexing(text, lang):
    return [lang.word2index.get(word, UNK_token) for word in text.split()]


def create_tensors(array, lang, max_len=MAX_LENGTH):
    input_tensors = []
    for sent in array:
        seq = [CLS_token] + indexing(sent, lang)
        seq = seq[:max_len]
        seq += [PAD_token] * (max_len - len(seq))
        input_tensors.append(seq)
    return torch.tensor(input_tensors, dtype=torch.long)


def build_vocab_and_sentences(sentences):
    lang = Input("sen")
    cleaned = sentences.fillna("").apply(normalizeString)
    inp = []
    for sentence in cleaned:
        if sentence.strip():
            lang.addSentence(sentence)
            inp.append(sentence)
    return lang, inp


class BertDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


def make_loader(input_tensors, labels, batch_size=BATCH_SIZE):
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)
    return DataLoader(BertDataset(input_tensors, labels), batch_size=batch_size, shuffle=True)
