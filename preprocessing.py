
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
from typing import List, Dict
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 use_stemming: bool = True,
                 use_lemmatization: bool = False,
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 min_token_length: int = 2):
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.min_token_length = min_token_length
        
        
        self._download_nltk_data()
        
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        
        logger.info("TextPreprocessor initialized with settings:")
        logger.info(f"  - Remove stopwords: {remove_stopwords}")
        logger.info(f"  - Stemming: {use_stemming}")
        logger.info(f"  - Lemmatization: {use_lemmatization}")
        logger.info(f"  - Lowercase: {lowercase}")
    
    def _download_nltk_data(self):
        required_data = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                try:
                    nltk.download(data, quiet=True)
                except:
                    logger.warning(f"Could not download {data}. Please run: python -m nltk.downloader {data}")
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        text = self.clean_text(text)
        if self.lowercase:
            text = text.lower()
        tokens = word_tokenize(text)
        if self.remove_punctuation:
            tokens = [token for token in tokens if token not in string.punctuation]
        tokens = [token for token in tokens if len(token) >= self.min_token_length]
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        if self.use_lemmatization and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens
    
    def preprocess(self, text: str) -> str:
        tokens = self.tokenize(text)
        return ' '.join(tokens)
    
    def preprocess_documents(self, documents: List[str]) -> List[List[str]]:
        return [self.tokenize(doc) for doc in documents]


def load_and_preprocess_data(filepath: str, 
                             text_column: str = 'Article',
                             preprocessor: TextPreprocessor = None) -> pd.DataFrame:
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    logger.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='cp1252')
    
    logger.info(f"Loaded {len(df)} documents")
    logger.info("Preprocessing documents...")
    df['tokens'] = df[text_column].apply(lambda x: preprocessor.tokenize(str(x)))
    df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))
    logger.info("Preprocessing complete")
    return df
