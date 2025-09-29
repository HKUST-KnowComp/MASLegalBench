import math

from six import iteritems
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25


class BM25(object):
    """Implementation of Best Matching 25 ranking function.

    Attributes
    ----------
    corpus_size : int
        Size of corpus (number of documents).
    avgdl : float
        Average length of document in `corpus`.
    doc_freqs : list of dicts of int
        Dictionary with terms frequencies for each document in `corpus`. Words used as keys and frequencies as values.
    idf : dict
        Dictionary with inversed documents frequencies for whole `corpus`. Words used as keys and frequencies as values.
    doc_len : list of int
        List of document lengths.
    """

    def __init__(self, corpus):
        """
        Parameters
        ----------
        corpus : list of list of str
            Given corpus.

        """
        self.corpus_size = 0
        self.corpus = corpus
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.nd = {}
        self._initialize(corpus)


    def _initialize(self, corpus):
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.corpus_size += 1
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        self.avgdl = float(num_doc) / self.corpus_size
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in iteritems(nd):
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = float(idf_sum) / len(self.idf)

        eps = EPSILON * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_score(self, document, index):
        """Computes BM25 score of given `document` in relation to item of corpus selected by `index`.

        Parameters
        ----------
        document : list of str
            Document to be scored.
        index : int
            Index of document in corpus selected to score with `document`.

        Returns
        -------
        float
            BM25 score.

        """
        score = 0
        doc_freqs = self.doc_freqs[index]
        for word in document:
            if word not in doc_freqs:
                continue
            score += (self.idf[word] * doc_freqs[word] * (PARAM_K1 + 1)
                      / (doc_freqs[word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.doc_len[index] / self.avgdl)))
            # score += self.idf[word] * doc_freqs[word]

        return score

    def get_scores(self, document):
        """Computes and returns BM25 scores of given `document` in relation to
        every item in corpus.

        Parameters
        ----------
        document : list of str
            Document to be scored.

        Returns
        -------
        list of float
            BM25 scores.

        """
        scores = [(self.get_score(document, index), index) for index in range(self.corpus_size)]
        return scores

    def get_words_score(self,document, index):

        words_score = {}
        doc_freqs = self.doc_freqs[index]
        for word in document:
            if word not in doc_freqs:
                continue
            score = (self.idf[word] * doc_freqs[word] * (PARAM_K1 + 1)
                      / (doc_freqs[word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.doc_len[index] / self.avgdl)))
            # score = self.idf[word] * doc_freqs[word]
            if word not in words_score:
                words_score[word] = score
            else:
                words_score[word] = max(words_score[word],score)
        word_score_tuples = [(word, score) for word, score in words_score.items()]
        word_score_tuples = sorted(word_score_tuples,key=lambda x:x[1],reverse=True)
        return word_score_tuples

    def get_most_relevant(self, query, num=5):
        result = []
        score_index = self.get_scores(query)
        score_index = sorted(score_index, key=lambda x: x[0], reverse=True)[:num]
        index = [si[1] for si in score_index]
        for idx in index:
            result.append(f"{self.corpus[idx]}")
        return result




class EMB(object):

    def __init__(self, corpus):

        self._initialize(corpus)


    def _initialize(self, corpus):
        self.corpus = corpus
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.corpus_size = len(corpus)
        self.emb = []

        with torch.no_grad():
            for doc in corpus:
                if isinstance(doc, list):
                    doc_text = " ".join(doc)
                else:
                    doc_text = str(doc)
                encoded_input = self.tokenizer(doc_text, padding=True, truncation=True, return_tensors='pt').to(self.device)
                model_output = self.model(**encoded_input)
                # 取[CLS]向量作为句子嵌入
                embeddings = model_output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                self.emb.append(embeddings)
        if self.corpus_size > 0:
            self.avgdl = sum(len(doc.split()) for doc in corpus) / self.corpus_size
        else:
            self.avgdl = 0

    def get_score(self, document, index):


        # 将document转为文本
        if isinstance(document, list):
            doc_text = " ".join(document)
        else:
            doc_text = str(document)
        # 计算document的嵌入
        with torch.no_grad():
            encoded_input = self.tokenizer(doc_text, padding=True, truncation=True, return_tensors='pt').to(self.device)
            model_output = self.model(**encoded_input)
            doc_emb = model_output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        corpus_emb = self.emb[index]
        if len(doc_emb.shape) > 1:
            doc_emb = doc_emb.reshape(-1)
        if len(corpus_emb.shape) > 1:
            corpus_emb = corpus_emb.reshape(-1)
        if np.linalg.norm(doc_emb) == 0 or np.linalg.norm(corpus_emb) == 0:
            return 0.0
        sim = np.dot(doc_emb, corpus_emb) / (np.linalg.norm(doc_emb) * np.linalg.norm(corpus_emb))
        return float(sim)


    def get_scores(self, document):
        """Computes and returns BM25 scores of given `document` in relation to
        every item in corpus.

        Parameters
        ----------
        document : list of str
            Document to be scored.

        Returns
        -------
        list of float
            BM25 scores.

        """
        scores = [(self.get_score(document, index), index) for index in range(self.corpus_size)]
        return scores



    def get_most_relevant(self, query, num=5):
        result = []
        score_index = self.get_scores(query)
        score_index = sorted(score_index, key=lambda x: x[0], reverse=True)[:num]
        index = [si[1] for si in score_index]
        for idx in index:
            result.append(f"{self.corpus[idx]}")
        return result