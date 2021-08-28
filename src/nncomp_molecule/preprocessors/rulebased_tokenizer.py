import json
import re
from pathlib import Path

from tqdm.auto import tqdm

import nncomp.registry as R


def split_form(form):
    string = ''
    for i in re.findall(r"[A-Z][^A-Z]*", form):
        elem = re.match(r"\D+", i).group()
        num = i.replace(elem, "")
        if num == "":
            string += f"{elem} "
        else:
            string += f"{elem} {str(num)} "
    return string.rstrip(' ')


def split_form2(form):
    string = ''
    for i in re.findall(r"[a-z][^a-z]*", form):
        elem = i[0]
        num = i.replace(elem, "").replace('/', "")
        num_string = ''
        for j in re.findall(r"[0-9]+[^0-9]*", num):
            num_list = list(re.findall(r'\d+', j))
            assert len(num_list) == 1
            _num = num_list[0]
            if j == _num:
                num_string += f"{_num} "
            else:
                extra = j.replace(_num, "")
                num_string += f"{_num} {' '.join(list(extra))} "
        string += f"/{elem} {num_string}"
    return string.rstrip(' ')


@R.PreprocessorRegistry.add
class InChIRuleBasedTokenizer:

    special_tokens = [
        "<PAD>", "<UNK>", "<BOS>", "<EOS>", "InChI=1S/"
    ]

    def __init__(self, config=None):
        self.stoi = {}
        self.itos = {}
        if config is not None:
            self.load(config)

    def __len__(self):
        return len(self.stoi)

    def save(self, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.stoi, f)

    def load(self, filepath):
        with open(filepath) as f:
            self.stoi = json.load(f)
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def preprocess(self, text):
        # text = split_form(text)

        text = split_form2(text)
        return text

    def fit_on_texts(self, texts):
        vocab = set()
        for text in tqdm(texts):
            text = self.preprocess(text)
            vocab.update(text.split(' '))
        vocab = sorted(vocab)
        vocab = [
            *self.special_tokens,
            *vocab,
        ]
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def __call__(self, text: str, text_pair: str = None):
        text = self.preprocess(text)
        token_ids = [
            self.stoi[token]
            for token in ["<BOS>", *text.split(" "), "<EOS>"]
        ]
        return dict(
            token_ids=token_ids,
            next_token_ids=[*token_ids[1:], 0]
        )
        return token_ids

    def token_to_id(self, token):
        return self.stoi[token]

    def decode(self, token_ids):
        text = ""
        for token_id in token_ids:
            token = self.itos[token_id]
            if token == "<EOS>":
                break
            if token_id >= len(self.special_tokens):
                text += token
        return text

    def decode_batch(self, list_of_token_ids):
        return [
            self.decode(token_ids)
            for token_ids in list_of_token_ids
        ]
