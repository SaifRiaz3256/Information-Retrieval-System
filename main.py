
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import logging
from typing import List, Dict

from preprocessing import TextPreprocessor, load_and_preprocess_data
from retrieval import RetrievalSystem
from evaluation import (
    RetrievalEvaluator, 
    create_synthetic_relevance_judgments,
    plot_evaluation_results
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IRSystemInterface:
    
    def __init__(self, 
                 data_path: str = "Articles.csv",
                 retrieval_method: str = "hybrid",
                 use_stemming: bool = True,
                 remove_stopwords: bool = True):
        logger.info("="*80)
        logger.info("INFORMATION RETRIEVAL SYSTEM - CS 516 Assignment 3")
        logger.info("="*80)
        
        logger.info("\n[1/3] Initializing text preprocessor...")
        self.preprocessor = TextPreprocessor(
            use_stemming=use_stemming,
            remove_stopwords=remove_stopwords,
            lowercase=True,
            remove_punctuation=True
        )
        
        logger.info("\n[2/3] Loading and preprocessing dataset...")
        self.df = load_and_preprocess_data(data_path, preprocessor=self.preprocessor)
        logger.info(f"Loaded {len(self.df)} documents")
        
        logger.info(f"\n[3/3] Building {retrieval_method.upper()} retrieval index...")
        self.system = RetrievalSystem(
            self.df, 
            retrieval_method=retrieval_method,
            preprocessor=self.preprocessor
        )
        
        logger.info("\n" + "="*80)
        logger.info("SYSTEM READY")
        logger.info("="*80 + "\n")
    
    def interactive_mode(self):
        print("\n" + "="*80)
        print("INTERACTIVE QUERY MODE")
        print("="*80)
        print("Enter your queries below. Type 'quit' or 'exit' to stop.")
        print("Type 'stats' to see system statistics.")
        print("="*80 + "\n")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if query.lower() == 'stats':
                    self._show_statistics()
                    continue
                
                results = self.system.query(query, top_k=10)
                
                self._display_results(query, results)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                continue
    
    def batch_mode(self, queries: List[str], output_file: str = None):
        print("\n" + "="*80)
        print(f"BATCH MODE - Processing {len(queries)} queries")
        print("="*80 + "\n")
        
        all_results = {}
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Query: {query}")
            print("-" * 80)
            
            results = self.system.query(query, top_k=10)
            
            all_results[query] = {
                'num_results': len(results),
                'top_results': results[['doc_id', 'score', 'Heading']].to_dict('records')
            }
            
            if len(results) > 0:
                print(f"\nTop 3 results:")
                for idx, row in results.head(3).iterrows():
                    print(f"  {idx+1}. [{row['score']:.4f}] {row['Heading']}")
            else:
                print("  No results found.")
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"\n\nResults saved to: {output_file}")
        
        return all_results
    
    def evaluate_mode(self, test_queries: List[str], output_dir: str = "evaluation_results"):
        print("\n" + "="*80)
        print("EVALUATION MODE")
        print("="*80 + "\n")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        evaluator = RetrievalEvaluator(self.system)
        
        print("Creating relevance judgments...")
        relevance_judgments = create_synthetic_relevance_judgments(
            self.df, test_queries, self.preprocessor
        )
        
        print("\nEvaluating queries...")
        eval_results = evaluator.evaluate_test_queries(
            test_queries, 
            relevance_judgments,
            k_values=[5, 10, 20]
        )
        
        eval_results.to_csv(f"{output_dir}/evaluation_results.csv", index=False)
        print(f"\nEvaluation results saved to: {output_dir}/evaluation_results.csv")
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print("\nAverage metrics by K:")
        summary = eval_results.groupby('k')[['precision', 'recall', 'f1']].mean()
        print(summary.to_string())
        
        print(f"\n\nAverage query time: {eval_results['query_time_ms'].mean():.2f} ms")
        
        memory = evaluator.measure_memory_usage()
        print(f"Memory usage (RSS): {memory['rss_mb']:.2f} MB")
        
        print("\nGenerating evaluation plots...")
        plot_evaluation_results(eval_results, output_dir)
        
        print(f"\nAll evaluation results saved to: {output_dir}/")
        
        return eval_results
    
    def _display_results(self, query: str, results: pd.DataFrame):
        print("\n" + "="*80)
        print(f"Results for: '{query}'")
        print("="*80)
        
        if len(results) == 0:
            print("\nNo results found.")
            return
        
        print(f"\nFound {len(results)} results:\n")
        
        for i, (_, row) in enumerate(results.iterrows(), 1):
            print(f"{i}. [Score: {row['score']:.4f}] {row['Heading']}")
            print(f"   Date: {row['Date']} | Type: {row['NewsType']}")
            
            article_text = row['Article'][:200].replace('\n', ' ')
            print(f"   {article_text}...")
            print()
    
    def _show_statistics(self):
        print("\n" + "="*80)
        print("SYSTEM STATISTICS")
        print("="*80)
        print(f"\nTotal documents: {len(self.df)}")
        print(f"Retrieval method: {self.system.retrieval_method}")
        
        vocab_size = len(set(token for tokens in self.df['tokens'] for token in tokens))
        print(f"Vocabulary size: {vocab_size}")
        
        avg_doc_length = self.df['tokens'].apply(len).mean()
        print(f"Average document length: {avg_doc_length:.2f} tokens")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Information Retrieval System - CS 516 Assignment 3"
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        default='Articles.csv',
        help='Path to the dataset (default: Articles.csv)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['boolean', 'tfidf', 'bm25', 'hybrid'],
        default='hybrid',
        help='Retrieval method (default: hybrid)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['interactive', 'batch', 'evaluate'],
        default='interactive',
        help='Operating mode (default: interactive)'
    )
    
    parser.add_argument(
        '--queries',
        type=str,
        nargs='+',
        help='Queries for batch/evaluate mode'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file/directory for results'
    )
    
    parser.add_argument(
        '--no-stemming',
        action='store_true',
        help='Disable stemming'
    )
    
    parser.add_argument(
        '--keep-stopwords',
        action='store_true',
        help='Keep stopwords (do not remove)'
    )
    
    args = parser.parse_args()
    
    try:
        ir_system = IRSystemInterface(
            data_path=args.data,
            retrieval_method=args.method,
            use_stemming=not args.no_stemming,
            remove_stopwords=not args.keep_stopwords
        )
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        sys.exit(1)
    
    try:
        if args.mode == 'interactive':
            ir_system.interactive_mode()
        
        elif args.mode == 'batch':
            if not args.queries:
                logger.error("Batch mode requires --queries argument")
                sys.exit(1)
            
            ir_system.batch_mode(args.queries, args.output)
        
        elif args.mode == 'evaluate':
            if not args.queries:
                test_queries = [
                    "oil prices crude market",
                    "stock market trading",
                    "economic growth GDP",
                    "government policy inflation",
                    "asian stocks hong kong"
                ]
            else:
                test_queries = args.queries
            
            output_dir = args.output or "evaluation_results"
            ir_system.evaluate_mode(test_queries, output_dir)
    
    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
