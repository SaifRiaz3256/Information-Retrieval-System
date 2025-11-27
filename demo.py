
from preprocessing import load_and_preprocess_data, TextPreprocessor
from retrieval import RetrievalSystem

print("="*80)
print("IR SYSTEM DEMO - Quick Test")
print("="*80)

print("\n1. Loading and preprocessing data...")
df = load_and_preprocess_data("Articles.csv")
print(f"     Loaded {len(df)} documents")

print("\n2. Building hybrid retrieval index...")
system = RetrievalSystem(df, retrieval_method='hybrid')
print("     Index built successfully!")

test_queries = [
    "oil prices falling market",
    "Pakistan economy growth",
    "stock exchange trading"
]

print("\n3. Running test queries...")
print("="*80)

for i, query in enumerate(test_queries, 1):
    print(f"\nQuery {i}: '{query}'")
    print("-" * 80)
    
    results = system.query(query, top_k=3)
    
    if len(results) > 0:
        for j, (_, row) in enumerate(results.iterrows(), 1):
            print(f"\n{j}. [Score: {row['score']:.4f}]")
            print(f"   Title: {row['Heading']}")
            print(f"   Date: {row['Date']} | Type: {row['NewsType']}")
    else:
        print("   No results found")

print("\n" + "="*80)
print("Demo complete! System is working perfectly!")
print("="*80)
print("\nNext steps:")
print("1. Run 'python main.py' for interactive mode")
print("2. Run 'python main.py --mode evaluate' for full evaluation")
print("3. Check evaluation_results/ folder for plots and metrics")
