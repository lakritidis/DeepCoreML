import time
import numpy as np

from sklearn.preprocessing import LabelEncoder

from Datasets import BaseDataset
from TextPreprocessor import TextPreprocessor


# #################################################################################################################
# TEXT DATASET DERIVED CLASS
class TextDataset(BaseDataset):
    """
    A TextDataset object contains two columns:
     * one column in a raw text format
     * one column with the respective class labels

    TextDataset inherits properties from BaseDataset.
    """
    def __init__(self, random_state=0):
        """
        """
        super().__init__(random_state)

        self.preprocessing_time_ = 0
        self.vectorization_time_ = 0
        self.projection_time_ = 0

    # Performs text preprocessing by using the TextPreprocessor object. Basic preprocessing takes place at word level.
    def preprocess(self):
        """
        Performs text preprocessing by using the TextPreprocessor object. Basic preprocessing takes place at word level.

        In both cases, the following four filters are applied:
         * tokenization
         * case folding,
         * punctuation removal and
         * stopword removal.
        """
        self.df_.columns.values[self.feature_columns_] = '_text_'
        self.df_.columns.values[self.class_column_] = '_class_'

        self.df_['_text_'].replace('', np.nan, inplace=True)
        self.df_.dropna(subset=['_text_'], inplace=True)

        t0 = time.time()
        preprocessor = TextPreprocessor()

        self.df_['_clean_text_'] = self.df_['_text_'].apply(lambda inp: preprocessor.preprocess_word(inp))
        self.df_['_clean_text_'].replace('', np.nan, inplace=True)
        self.df_.dropna(subset=['_clean_text_'], inplace=True)

        self.x_ = self.df_['_clean_text_']
        self.y_ = self.df_['_class_']

        class_encoder = LabelEncoder()
        self.y_ = class_encoder.fit_transform(self.y_)

        self.preprocessing_time_ = time.time() - t0

    # Generate text vectors of the text column.
    def vectorize(self, vectorizer):
        """
        Generates text vectors of the text column by using the specified vectorizer object. This method calls the
        preprocess() method internally and computes the text vectors of the _clean_text column.

        :param vectorizer: The text vectorization method (An object of type TextVectorizer from TextVectorizers.py)
        :return: A numpy array with the text vectors
        """
        print("\tText Vectorization with", vectorizer, "... ", end="", flush=True)

        raw_data = self.df_['_clean_text_'].to_numpy()

        vector_data = vectorizer.fit_transform(raw_data)
        self.vectorization_time_ = vectorizer.vectorization_time_
        self.dimensionality_ = vectorizer.get_dimensionality()

        print("completed in %5.2f sec (dimensionality = %d)." % (self.vectorization_time_, self.dimensionality_))
        return vector_data

    def get_preprocessing_time(self):
        """
        :return: The dataset preprocessing dataset
        """
        return self.preprocessing_time_
