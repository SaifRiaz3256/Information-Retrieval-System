
print("="*80)
print("EXAMPLE 1: Basic Interactive Query")
print("="*80)

from preprocessing import load_and_preprocess_data, TextPreprocessor
from retrieval import RetrievalSystem

print("\n1. Loading data...")
df = load_and_preprocess_data("Articles.csv")
print(f"   Loaded {len(df)} documents")

print("\n2. Building retrieval index...")
system = RetrievalSystem(df, retrieval_method='hybrid')
print("   Index built successfully!")

print("\n3. Executing query...")
query = "oil prices falling stock market"
results = system.query(query, top_k=5)

print(f"\n4. Results for: '{query}'")
print("-" * 80)
for i, (_, row) in enumerate(results.iterrows(), 1):
    print(f"\n{i}. [Score: {row['score']:.4f}]")
    print(f"   Title: {row['Heading']}")
    print(f"   Date: {row['Date']} | Type: {row['NewsType']}")
    article_snippet = row['Article'][:150].replace('\n', ' ')
    print(f"   {article_snippet}...")

print("\n\n" + "="*80)
print("EXAMPLE 2: Comparing Retrieval Methods")
print("="*80)

query = "economic growth inflation"
print(f"\nQuery: '{query}'")

methods = ['tfidf', 'bm25', 'hybrid']
for method in methods:
    print(f"\n{method.upper()} Method:")
    print("-" * 40)
    
    system = RetrievalSystem(df, retrieval_method=method)
    results = system.query(query, top_k=3)
    
    for i, (_, row) in enumerate(results.iterrows(), 1):
        print(f"  {i}. [{row['score']:.4f}] {row['Heading'][:60]}...")

print("\n\n" + "="*80)
print("EXAMPLE 3: System Evaluation")
print("="*80)

from evaluation import RetrievalEvaluator, create_synthetic_relevance_judgments

evaluator = RetrievalEvaluator(system)

test_queries = [
    "oil prices market",
    "stock exchange trading"
]

print("\nTest queries:", test_queries)

preprocessor = TextPreprocessor()
relevance_judgments = create_synthetic_relevance_judgments(df, test_queries, preprocessor)

print("\nRunning evaluation...")
eval_results = evaluator.evaluate_test_queries(test_queries, relevance_judgments, k_values=[5, 10])

print("\nEvaluation Results:")
print(eval_results.groupby('k')[['precision', 'recall', 'f1']].mean())

memory = evaluator.measure_memory_usage()
print(f"\nMemory Usage: {memory['rss_mb']:.2f} MB")

print("\n" + "="*80)
print("Examples complete! See main.py for full interface.")
print("="*80)
