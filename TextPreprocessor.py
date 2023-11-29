import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup


class TextPreprocessor:
    """
    TextPreprocessor class performs basic text operations including case-folding, lemmatixation, punctuation removal
    and stopword removal
    """
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

        self.lemmatizer_ = WordNetLemmatizer()
        self.stopwords_ = set(stopwords.words('english'))

    def preprocess_word(self, text):
        """
        Receives raw un-formatted text and strips punctuation, numbers, and whitespaces. Then, it tokenizes each input
        string and stems the generated words.
        :param text: input text in raw format to be processed.
        :return:
        """
        # Lowercase and strip everything except words
        cleaner = re.sub(r"[^a-zA-Z ]+", ' ', str(text).lower())

        # Tokenize
        cleaner = word_tokenize(cleaner)
        clean = []
        for w in cleaner:
            # filter out stopwords
            if w not in self.stopwords_:
                # filter out short words
                if len(w) > 2:
                    clean.append(self.lemmatizer_.lemmatize(w))
        return ' '.join(clean)

    def preprocess_wordlist(self, text):
        """

        :param text:
        :return:
        """
        # Remove HTML tag
        review = BeautifulSoup(text, 'html.parser').get_text()

        # Remove non-letters
        review = re.sub('[^a-zA-Z]', ' ', review)

        # Convert to lower case
        review = review.lower()

        # Tokenize
        word = nltk.word_tokenize(review)

        # Optional: Remove stop words (false by default)
        words = [w for w in word if w not in self.stopwords_]

        return words

    def preprocess_sent(self, text):
        # Split the paragraph into sentences

        # raw = tokenizer.tokenize(data.strip())
        raw = nltk.sent_tokenize(str(text).strip())

        # If the length of the sentence is greater than 0, plug the sentence in the function preprocess_wordlist
        sentences = [self.preprocess_wordlist(sent) for sent in raw if len(sent) > 0]

        return sentences
