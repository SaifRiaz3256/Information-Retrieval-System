
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
import time
import psutil
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    
    def __init__(self, retrieval_system):
        self.system = retrieval_system
        self.process = psutil.Process(os.getpid())
    
    def precision_at_k(self, retrieved: List[int], relevant: Set[int], k: int) -> float:
        if k == 0 or not retrieved:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
        
        return relevant_retrieved / k
    
    def recall_at_k(self, retrieved: List[int], relevant: Set[int], k: int) -> float:
        if not relevant or not retrieved:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
        
        return relevant_retrieved / len(relevant)
    
    def f1_score(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def average_precision(self, retrieved: List[int], relevant: Set[int]) -> float:
        if not relevant or not retrieved:
            return 0.0
        
        precision_sum = 0.0
        relevant_retrieved = 0
        
        for k, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                relevant_retrieved += 1
                precision_sum += relevant_retrieved / k
        
        if relevant_retrieved == 0:
            return 0.0
        
        return precision_sum / len(relevant)
    
    def mean_average_precision(self, 
                               queries: List[str],
                               relevance_judgments: Dict[str, Set[int]]) -> float:
        ap_scores = []
        
        for query in queries:
            results = self.system.query(query, top_k=100)
            retrieved = results['doc_id'].tolist()
            relevant = relevance_judgments.get(query, set())
            
            ap = self.average_precision(retrieved, relevant)
            ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def dcg_at_k(self, retrieved: List[int], relevance_scores: Dict[int, float], k: int) -> float:
        if not retrieved:
            return 0.0
        
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += rel / np.log2(i + 1)
        
        return dcg
    
    def ndcg_at_k(self, retrieved: List[int], relevance_scores: Dict[int, float], k: int) -> float:
        dcg = self.dcg_at_k(retrieved, relevance_scores, k)
        
        ideal_order = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_order))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def measure_query_time(self, query: str, iterations: int = 10) -> Dict[str, float]:
        times = []
        
        for _ in range(iterations):
            start = time.time()
            self.system.query(query, top_k=10)
            end = time.time()
            times.append(end - start)
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
    
    def measure_memory_usage(self) -> Dict[str, float]:
        mem_info = self.process.memory_info()
        
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024),
        }
    
    def evaluate_test_queries(self, 
                             test_queries: List[str],
                             relevance_judgments: Dict[str, Set[int]],
                             k_values: List[int] = [5, 10, 20]) -> pd.DataFrame:
        results = []
        
        for query in test_queries:
            logger.info(f"Evaluating query: '{query}'")
            
            start = time.time()
            search_results = self.system.query(query, top_k=max(k_values))
            query_time = time.time() - start
            
            retrieved = search_results['doc_id'].tolist()
            relevant = relevance_judgments.get(query, set())
            
            for k in k_values:
                precision = self.precision_at_k(retrieved, relevant, k)
                recall = self.recall_at_k(retrieved, relevant, k)
                f1 = self.f1_score(precision, recall)
                
                results.append({
                    'query': query,
                    'k': k,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'retrieved_count': min(len(retrieved), k),
                    'relevant_count': len(relevant),
                    'query_time_ms': query_time * 1000
                })
        
        return pd.DataFrame(results)


def create_synthetic_relevance_judgments(df: pd.DataFrame,
                                        queries: List[str],
                                        preprocessor) -> Dict[str, Set[int]]:
    relevance_judgments = {}
    
    for query in queries:
        query_tokens = set(preprocessor.tokenize(query))
        relevant_docs = set()
        
        for idx, row in df.iterrows():
            doc_tokens = set(row['tokens'])
            
            overlap = len(query_tokens.intersection(doc_tokens))
            overlap_ratio = overlap / len(query_tokens) if query_tokens else 0
            
            if overlap >= 2 or overlap_ratio >= 0.4:
                relevant_docs.add(idx)
        
        relevance_judgments[query] = relevant_docs
        logger.info(f"Query '{query}': {len(relevant_docs)} relevant documents")
    
    return relevance_judgments


def plot_evaluation_results(eval_df: pd.DataFrame, output_dir: str = "."):
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ['precision', 'recall', 'f1']
    titles = ['Precision@K', 'Recall@K', 'F1@K']
    
    for ax, metric, title in zip(axes, metrics, titles):
        grouped = eval_df.groupby('k')[metric].mean()
        ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('K', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_by_k.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    if len(eval_df['query'].unique()) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        plot_data = eval_df[eval_df['k'] == 10].copy()
        plot_data = plot_data.sort_values('f1', ascending=True)
        
        ax.barh(range(len(plot_data)), plot_data['f1'])
        ax.set_yticks(range(len(plot_data)))
        ax.set_yticklabels(plot_data['query'], fontsize=10)
        ax.set_xlabel('F1 Score', fontsize=12)
        ax.set_title('F1@10 by Query', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/f1_by_query.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Evaluation plots saved to {output_dir}")


if __name__ == "__main__":
    from preprocessing import load_and_preprocess_data, TextPreprocessor
    from retrieval import RetrievalSystem
    
    df = load_and_preprocess_data("Articles.csv")
    
    system = RetrievalSystem(df[:1000], retrieval_method='hybrid')
    
    test_queries = [
        "oil prices market",
        "stock exchange trading",
        "economic growth inflation"
    ]
    
    evaluator = RetrievalEvaluator(system)
    
    preprocessor = TextPreprocessor()
    relevance_judgments = create_synthetic_relevance_judgments(
        df[:1000], test_queries, preprocessor
    )
    
    results = evaluator.evaluate_test_queries(test_queries, relevance_judgments)
    print("\nEvaluation Results:")
    print(results)
    
    plot_evaluation_results(results)
