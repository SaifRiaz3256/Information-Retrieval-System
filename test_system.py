
import sys
import os

def test_imports():
    
    print("Testing imports...")
    try:
        import pandas
        import numpy
        import sklearn
        import nltk
        import rank_bm25
        import matplotlib
        import seaborn
        print(" All dependencies imported successfully")
        return True
    except ImportError as e:
        print(f" Import error: {e}")
        return False

def test_preprocessing():
    
    print("\nTesting preprocessing module...")
    try:
        from preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        sample_text = "This is a TEST! It should be processed correctly."
        
        tokens = preprocessor.tokenize(sample_text)
        processed = preprocessor.preprocess(sample_text)
        
        assert len(tokens) > 0, "No tokens generated"
        assert len(processed) > 0, "No processed text generated"
        
        print(f"  Original: {sample_text}")
        print(f"  Tokens: {tokens}")
        print(f"  Processed: {processed}")
        print("  Preprocessing module working")
        return True
    except Exception as e:
        print(f"  Preprocessing error: {e}")
        return False

def test_data_loading():
    print("\nTesting retrieval system...")
    try:
        from preprocessing import load_and_preprocess_data, TextPreprocessor
        from retrieval import RetrievalSystem
        
        
        import pandas as pd
        try:
            df = pd.read_csv("Articles.csv", nrows=100, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv("Articles.csv", nrows=100, encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv("Articles.csv", nrows=100, encoding='cp1252')
        
        preprocessor = TextPreprocessor()
        df['processed_text'] = df['Article'].fillna('').apply(preprocessor.preprocess)
        df['tokens'] = df['Article'].fillna('').apply(preprocessor.tokenize)
        df['doc_length'] = df['tokens'].apply(len)
        
        
        for method in ['tfidf', 'bm25', 'hybrid']:
            print(f"  Testing {method.upper()} retrieval...")
            system = RetrievalSystem(df, retrieval_method=method, preprocessor=preprocessor)
            results = system.query("oil market prices", top_k=5)
            assert len(results) <= 5, f"Too many results returned by {method}"
            print(f"      {method.upper()} returned {len(results)} results")
        
        print("  Retrieval system working")
        return True
    except Exception as e:
        print(f"  Retrieval error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation():
    
    print("\nTesting evaluation module...")
    try:
        from preprocessing import TextPreprocessor
        from retrieval import RetrievalSystem
        from evaluation import RetrievalEvaluator
        import pandas as pd
        
        
        try:
            df = pd.read_csv("Articles.csv", nrows=100, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv("Articles.csv", nrows=100, encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv("Articles.csv", nrows=100, encoding='cp1252')
        
        preprocessor = TextPreprocessor()
        df['processed_text'] = df['Article'].fillna('').apply(preprocessor.preprocess)
        df['tokens'] = df['Article'].fillna('').apply(preprocessor.tokenize)
        df['doc_length'] = df['tokens'].apply(len)
        
        system = RetrievalSystem(df, retrieval_method='hybrid', preprocessor=preprocessor)
        evaluator = RetrievalEvaluator(system)
        
        
        retrieved = [0, 1, 2, 3, 4]
        relevant = {1, 3, 5}
        
        precision = evaluator.precision_at_k(retrieved, relevant, 5)
        recall = evaluator.recall_at_k(retrieved, relevant, 5)
        
        print(f"  Precision@5: {precision:.3f}")
        print(f"  Recall@5: {recall:.3f}")
        
        print("  Evaluation module working")
        return True
    except Exception as e:
        print(f"  Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*80)
    print("IR SYSTEM TEST SUITE")
    print("="*80)
    
    tests = [
        ("Import Dependencies", test_imports),
        ("Preprocessing", test_preprocessing),
        ("Data Loading", test_data_loading),
        ("Evaluation Module", test_evaluation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  Unexpected error in {name}: {e}")
            results.append((name, False))
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "  PASS" if result else "  FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "="*80)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready for use.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please fix issues before using.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
