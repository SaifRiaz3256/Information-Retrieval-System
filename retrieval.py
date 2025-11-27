
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import logging
from preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BooleanRetrieval:
    
    def __init__(self):
        self.inverted_index = defaultdict(set)
        self.documents = []
        self.doc_ids = []
    
    def build_index(self, documents: List[List[str]], doc_ids: List[int]):
        logger.info("Building Boolean inverted index...")
        self.documents = documents
        self.doc_ids = doc_ids
        
        for doc_id, tokens in zip(doc_ids, documents):
            for token in set(tokens):
                self.inverted_index[token].add(doc_id)
        
        logger.info(f"Inverted index built with {len(self.inverted_index)} unique terms")
    
    def search(self, query_tokens: List[str], operator: str = 'OR') -> Set[int]:
        if not query_tokens:
            return set()
        
        result = self.inverted_index.get(query_tokens[0], set()).copy()
        
        for term in query_tokens[1:]:
            postings = self.inverted_index.get(term, set())
            if operator.upper() == 'AND':
                result = result.intersection(postings)
            elif operator.upper() == 'OR':
                result = result.union(postings)
        
        return result


class TFIDFRetrieval:
    
    def __init__(self, max_features: int = None, ngram_range: Tuple[int, int] = (1, 1)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=False,
            token_pattern=r'(?u)\b\w+\b'
        )
        self.doc_vectors = None
        self.doc_ids = []
    
    def build_index(self, documents: List[str], doc_ids: List[int]):
        logger.info("Building TF-IDF index...")
        self.doc_ids = doc_ids
        self.doc_vectors = self.vectorizer.fit_transform(documents)
        
        logger.info(f"TF-IDF matrix shape: {self.doc_vectors.shape}")
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        query_vector = self.vectorizer.transform([query])
        
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (self.doc_ids[idx], similarities[idx]) 
            for idx in top_indices 
            if similarities[idx] > 0
        ]
        
        return results


class BM25Retrieval:
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.doc_ids = []
    
    def build_index(self, documents: List[List[str]], doc_ids: List[int]):
        logger.info("Building BM25 index...")
        self.doc_ids = doc_ids
        self.bm25 = BM25Okapi(documents, k1=self.k1, b=self.b)
        
        logger.info(f"BM25 index built for {len(documents)} documents")
    
    def search(self, query_tokens: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            (self.doc_ids[idx], scores[idx]) 
            for idx in top_indices 
            if scores[idx] > 0
        ]
        
        return results


class HybridRetrieval:
    
    def __init__(self, 
                 tfidf_weight: float = 0.4,
                 bm25_weight: float = 0.6):
        self.tfidf_retrieval = TFIDFRetrieval()
        self.bm25_retrieval = BM25Retrieval()
        self.boolean_retrieval = BooleanRetrieval()
        
        self.tfidf_weight = tfidf_weight
        self.bm25_weight = bm25_weight
        
        total_weight = tfidf_weight + bm25_weight
        self.tfidf_weight /= total_weight
        self.bm25_weight /= total_weight
        
        logger.info(f"Hybrid retrieval initialized with weights:")
        logger.info(f"  TF-IDF: {self.tfidf_weight:.2f}")
        logger.info(f"  BM25: {self.bm25_weight:.2f}")
    
    def build_index(self, documents_tokens: List[List[str]], 
                   documents_text: List[str], 
                   doc_ids: List[int]):
        self.tfidf_retrieval.build_index(documents_text, doc_ids)
        
        self.bm25_retrieval.build_index(documents_tokens, doc_ids)
        
        self.boolean_retrieval.build_index(documents_tokens, doc_ids)
    
    def search(self, 
              query_text: str,
              query_tokens: List[str],
              top_k: int = 10,
              use_boolean_filter: bool = False) -> List[Tuple[int, float]]:
        tfidf_results = self.tfidf_retrieval.search(query_text, top_k=top_k*2)
        bm25_results = self.bm25_retrieval.search(query_tokens, top_k=top_k*2)
        
        combined_scores = defaultdict(float)
        
        if tfidf_results:
            max_tfidf = max(score for _, score in tfidf_results) if tfidf_results else 1.0
            for doc_id, score in tfidf_results:
                normalized_score = score / max_tfidf if max_tfidf > 0 else 0
                combined_scores[doc_id] += self.tfidf_weight * normalized_score
        
        if bm25_results:
            max_bm25 = max(score for _, score in bm25_results) if bm25_results else 1.0
            for doc_id, score in bm25_results:
                normalized_score = score / max_bm25 if max_bm25 > 0 else 0
                combined_scores[doc_id] += self.bm25_weight * normalized_score
        
        if use_boolean_filter:
            boolean_results = self.boolean_retrieval.search(query_tokens, operator='OR')
            combined_scores = {
                doc_id: score 
                for doc_id, score in combined_scores.items() 
                if doc_id in boolean_results
            }
        
        sorted_results = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        return sorted_results


class RetrievalSystem:
    
    def __init__(self, 
                 df: pd.DataFrame,
                 retrieval_method: str = 'hybrid',
                 preprocessor: TextPreprocessor = None):
        self.df = df
        self.retrieval_method = retrieval_method
        self.preprocessor = preprocessor or TextPreprocessor()
        
        if retrieval_method == 'boolean':
            self.engine = BooleanRetrieval()
        elif retrieval_method == 'tfidf':
            self.engine = TFIDFRetrieval()
        elif retrieval_method == 'bm25':
            self.engine = BM25Retrieval()
        elif retrieval_method == 'hybrid':
            self.engine = HybridRetrieval()
        else:
            raise ValueError(f"Unknown retrieval method: {retrieval_method}")
        
        self._build_index()
    
    def _build_index(self):
        logger.info(f"Building index using {self.retrieval_method} method...")
        
        doc_ids = self.df.index.tolist()
        
        if self.retrieval_method == 'hybrid':
            self.engine.build_index(
                documents_tokens=self.df['tokens'].tolist(),
                documents_text=self.df['processed_text'].tolist(),
                doc_ids=doc_ids
            )
        elif self.retrieval_method in ['boolean', 'bm25']:
            self.engine.build_index(
                documents=self.df['tokens'].tolist(),
                doc_ids=doc_ids
            )
        elif self.retrieval_method == 'tfidf':
            self.engine.build_index(
                documents=self.df['processed_text'].tolist(),
                doc_ids=doc_ids
            )
        
        logger.info("Index built successfully")
    
    def query(self, query_text: str, top_k: int = 10) -> pd.DataFrame:
        query_tokens = self.preprocessor.tokenize(query_text)
        query_processed = ' '.join(query_tokens)
        
        if self.retrieval_method == 'boolean':
            doc_ids = self.engine.search(query_tokens)
            results = [(doc_id, 1.0) for doc_id in doc_ids]
        elif self.retrieval_method == 'tfidf':
            results = self.engine.search(query_processed, top_k=top_k)
        elif self.retrieval_method == 'bm25':
            results = self.engine.search(query_tokens, top_k=top_k)
        elif self.retrieval_method == 'hybrid':
            results = self.engine.search(
                query_text=query_processed,
                query_tokens=query_tokens,
                top_k=top_k
            )
        
        if not results:
            return pd.DataFrame()
        
        result_doc_ids = [doc_id for doc_id, _ in results]
        result_scores = [score for _, score in results]
        
        result_df = self.df.loc[result_doc_ids].copy()
        result_df['score'] = result_scores
        result_df = result_df.sort_values('score', ascending=False)
        
        return result_df
