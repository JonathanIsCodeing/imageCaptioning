from collections import Counter

import nltk
import torch
import tqdm


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.start_word = '<start>'
        self.end_word = '<end>'
        self.pad_word = '<pad>'
        self.unk_word = '<unk>'

        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.pad_word)
        self.add_word(self.unk_word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def build(self, captions, threshold=3):
        counter = Counter()
        for img_captions in tqdm.tqdm(captions, desc="Expand Vocabulary: "):  # All captions of an image
            for caption in img_captions:
                tokens = nltk.tokenize.word_tokenize(caption.lower())
                counter.update(tokens)

        # Add word to vocab if counter is higher than threshold
        for word, count in counter.items():
            if count >= threshold:
                self.add_word(word)

    def tokenize_caption(self, caption, max_len):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokenized_caption = [self(self.start_word)]
        tokenized_caption.extend([self(token) for token in tokens])
        tokenized_caption.append(self(self.end_word))
        while len(tokenized_caption) < max_len:
            tokenized_caption.append(self(self.pad_word))

        tokenized_caption = torch.Tensor(tokenized_caption).long()

        return tokenized_caption

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return self.idx + 1
