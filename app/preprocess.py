import numpy as np
import nltk
import spacy
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from random import shuffle


class PreProcessor:
    def __init__(self) -> None:
        self.tokenizer = RegexpTokenizer("\/|^\.|\.$|,|;|\(|\)|^\-|\-$|:|;", gaps=True)
        self.ncol = 100
        self.cv_length = 500
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.stopset = dict(
            zip(stopwords.words("english"), range(len(stopwords.words("english"))))
        )
        self.wv_file = "app/data/vectors.kv"
        self.wv = KeyedVectors.load(self.wv_file, mmap="r")
        self.wv.init_sims()

    def clean(self, tokens):
        excluded = set(string.punctuation)
        excluded_2 = set(["year", "years", "etc", "#", "&"])
        import re

        url_pattern = re.compile(
            "^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?"
        )
        output = []
        for token in tokens:
            if (
                token not in excluded
                and token not in excluded_2
                and not url_pattern.match(token)
            ):
                for word in self.tokenizer.tokenize(token):
                    if len(word) >= 2:
                        output.append(word)

        return output

    def text_to_matrix(self, text: str):
        matrix = np.empty((0, self.ncol), dtype=np.float32)
        cv_sents = nltk.sent_tokenize(text)
        cv_tokens = [
            token.lemma_
            for sent in cv_sents
            for token in self.nlp(
                " ".join(self.clean(nltk.tokenize.word_tokenize(sent.lower())))
            )
        ]
        shuffle(cv_tokens)

        cpt = 0

        for word in cv_tokens:
            if word in self.wv and word not in self.stopset:
                vect = self.wv.word_vec(word, norm=True)
                matrix = np.append(
                    matrix, np.array([vect], dtype=np.float32), axis=0
                )
                cpt += 1
                if cpt == self.cv_length:
                    break

        for j in range(self.cv_length - cpt):
            matrix = np.append(
                matrix, np.array([[0] * self.ncol], dtype=np.float32), axis=0
            )

        return matrix
