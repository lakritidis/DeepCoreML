import time
import numpy as np
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec

from transformers import BertTokenizer, BertModel, BertConfig

from TextPreprocessor import TextPreprocessor


# ####################################################################################################################
# BaseTextVectorizer #################################################################################################
class BaseTextVectorizer:
    """
    Base class for text vectorization.
    """
    def __init__(self, latent_dimensionality):
        """
        :param latent_dimensionality: The length of text vectors generated by the vectorizer
        """
        self.vectorization_time_ = 0
        self.latent_dim_ = latent_dimensionality
        self.vectorizer_ = None

    def get_time(self):
        return self.vectorization_time_

    def get_dimensionality(self):
        return self.latent_dim_


# ####################################################################################################################
# tfidfVectorizer ####################################################################################################
class tfidfVectorizer(BaseTextVectorizer):
    """
    Wrapper class for sklearn's tfIdfvectorizer. This vectorizer is enforced to produce vectors with a fixed
    number of components
    """
    def __init__(self, latent_dimensionality=512, **kwargs):
        """
        Initialize a class to perform tf-idf text vectorization.
        :param latent_dimensionality: The size of the produced vectors (i.e. number of components)
        """
        super().__init__(latent_dimensionality)
        self.vectorizer_ = TfidfVectorizer(max_features=latent_dimensionality, **kwargs)

    def fit(self, x):
        """
        A simple invocation to the fit method of sklearn's tfIdfvectorizer
        :param x: Input raw text to be used for 'training' a tfIdfvectorizer
        """
        t0 = time.time()
        self.vectorizer_.fit(x)
        self.vectorization_time_ = time.time() - t0

    def transform(self, x):
        """
        A simple invocation to the transform method of sklearn's tfIdfvectorizer
        :param x: Input raw text to be transformed by the tfIdfvectorizer
        :return: text vectors of length self.latent_dimensionality_
        """
        return self.vectorizer_.transform(x)

    def fit_transform(self, x, y=None):
        self.vectorizer_.fit(x)
        x_vec = self.vectorizer_.transform(x)
        return x_vec


# ####################################################################################################################
# word2vecVectorizer #################################################################################################
class word2vecVectorizer(BaseTextVectorizer):
    def __init__(self, latent_dimensionality=512):
        super().__init__(latent_dimensionality)
        self.vectorizer_ = None
        self.preprocessor_ = TextPreprocessor()

    def fit(self, x, y=None):
        t0 = time.time()

        s_x = x.apply(lambda inp: self.preprocessor_.preprocess_sent(inp))

        sentence = []
        for s in s_x:
            sentence += s

        avg_sentence_length = np.average([len(s) for s in sentence])
        context = int(avg_sentence_length)

        min_count, num_processor, down_sampling = 1, 2, 0.001

        self.vectorizer_ = word2vec.Word2Vec(sentence, workers=num_processor, vector_size=self.latent_dim_,
                                             min_count=min_count, window=context, sample=down_sampling)
        self.vectorizer_.init_sims(replace=True)

        self.vectorization_time_ = time.time() - t0

    def transform(self, x):
        s_x = x.apply(lambda inp: self.preprocessor_.preprocess_sent(inp))

        data_vec = []
        ctr = -1
        for t in s_x:
            ctr += 1
            data_vec.append([])
            for s in t:
                for w in s:
                    data_vec[ctr].append(w)

        return self.get_avg_feature_vec(data_vec, self.vectorizer_)

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    def make_feature_vec(self, doc, model):
        feature_vec = np.zeros((self.latent_dim_,), dtype="float32")

        # Unique word set
        word_index = set(model.wv.index_to_key)
        # For division, we need to count the number of words
        num_words = 0

        # Iterate words in a review and if the word is in the unique word set, add the vector values for each word.
        for word in doc:
            if word in word_index:
                num_words += 1
                feature_vec = np.add(feature_vec, model.wv[word])
            # else:
                # print("word:", word, "not in index\n")

        # if  num_words == 0:
        #    print("Zero words! in", doc)
        # Divide the sum of vector values by total number of word in a review.
        # print(feature_vec)
        if num_words > 0:
            feature_vec = np.divide(feature_vec, num_words)

        return feature_vec

    def get_avg_feature_vec(self, clean_docs, model):
        # Keep track of the sequence of reviews, create the number "th" variable.
        review_th = 0

        # Row: number of total reviews, Column: number of vector spaces (num_features, we set this in Word2Vec step).
        feature_vecs = np.zeros((len(clean_docs), self.latent_dim_), dtype="float32")

        # Iterate over reviews and add the result of makeFeatureVec.
        for d in clean_docs:
            if len(d) > 0:
                vec = self.make_feature_vec(d, model)
                if vec.any():
                    feature_vecs[int(review_th)] = vec
                    # Once the vector values are added, increase the one for the review_th variable.
                    review_th += 1

        return feature_vecs


# ####################################################################################################################
# bertVectorizer #####################################################################################################
class bertVectorizer(BaseTextVectorizer):
    def __init__(self, model='google/bert_uncased_L-2_H-128_A-2', latent_dimensionality=128):
        super().__init__(latent_dimensionality)
        self.algorithm_ = model

        self.attention_heads_ = 12
        self.int_size_ = 3072

        if self.latent_dim_ == 128:
            self.attention_heads_ = 2
            self.int_size_ = 512
        elif self.latent_dim_ == 256:
            self.attention_heads_ = 4
            self.int_size_ = 1024
        elif self.latent_dim_ == 512:
            self.attention_heads_ = 8
            self.int_size_ = 2048

    def fit(self, x, y=None):
        t0 = time.time()

        # Set up the transformer's configuration for BERT fine-tuning
        dropout, attention_dropout = 0.2, 0.2
        bert_configuration = BertConfig(dropout=dropout, attention_dropout=attention_dropout, num_hidden_layers=12,
                                        output_hidden_states=True, hidden_size=self.latent_dim_,
                                        num_attention_heads=self.attention_heads_, intermediate_size=self.int_size_)

        # Create the models for the training and test sets
        self.vectorizer_ = BertModel.from_pretrained(self.algorithm_, config=bert_configuration)

        self.vectorization_time_ = time.time() - t0

    def transform(self, x):
        tokenizer = BertTokenizer.from_pretrained(self.algorithm_)
        encoded_input = tokenizer(x.tolist(), padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            bert_model = self.vectorizer_(**encoded_input)

        return self.mean_pooling(bert_model, encoded_input['attention_mask']).numpy()

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
