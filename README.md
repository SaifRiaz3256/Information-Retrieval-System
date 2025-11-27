##  Information Retrieval System

This is a complete, locally-running Information Retrieval (IR) system that implements multiple retrieval strategies and provides comprehensive evaluation metrics. The system processes 2,692 news articles and supports natural language queries with fast response times (~18ms average).

### Key Statistics
- **Dataset:** 2,692 news articles
- **Vocabulary:** 43,636 unique terms
- **Query Time:** ~18.83 ms average
- **Memory Usage:** ~271 MB
- **Precision@10:** 1.000
- **Indexing Time:** ~5 seconds

---


##  System Requirements

- **Python:** 3.8 or higher (tested with 3.13.1)
- **RAM:** Minimum 4GB recommended
- **Disk Space:** ~500MB for dependencies
- **OS:** Windows, macOS, or Linux

---

##  Installation

### Step 1: Navigate to Project Directory
```powershell
cd "Project path"
```

### Step 2: Create Virtual Environment
```powershell
python -m venv venv
```

### Step 3: Activate Virtual Environment

**On Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**On Windows Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

This installs:
- pandas 2.3.3
- numpy 2.3.5
- scikit-learn 1.7.2
- nltk 3.9.2
- rank-bm25 0.2.2
- matplotlib 3.10.7
- seaborn 0.13.2
- tqdm 4.67.1
- psutil 7.1.3

### Step 5: Download NLTK Data
```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

### Step 6: Verify Installation
```powershell
python test_system.py
```

You should see:
```
Results: 6/6 tests passed
All tests passed! System is ready for use.
```

---

##  Usage

### Quick Demo
```powershell
python demo.py
```
Shows example queries with results.

### Interactive Mode (Default)
```powershell
python main.py
```

Then enter queries:
```
Example:
Query: oil prices falling stock market
Query: Pakistan economy growth
Query: quit
```

### Batch Mode
```powershell
python main.py --mode batch --queries "oil prices" "stock market" "economic growth" --output results.json
```

### Evaluation Mode
```powershell
python main.py --mode evaluate --output evaluation_results
```

Generates:
- `evaluation_results/evaluation_results.csv` - Detailed metrics
- `evaluation_results/metrics_by_k.png` - Precision/Recall/F1 plots
- `evaluation_results/f1_by_query.png` - Per-query performance

### Different Retrieval Methods
```powershell

python main.py --method tfidf


python main.py --method bm25


python main.py --method boolean


python main.py --method hybrid
```

### Preprocessing Options
```powershell

python main.py --no-stemming


python main.py --keep-stopwords


python main.py --method bm25 --no-stemming
```


##  Project Structure

```
Assigment # 3/
├── Articles.csv                 # Dataset (2,692 news articles)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── TECHNICAL_DOCUMENTATION.docx # Comprehensive technical report
│
├── preprocessing.py             # Text preprocessing module
├── retrieval.py                 # Retrieval algorithms
├── evaluation.py                # Evaluation metrics
├── main.py                      # Main interface
├── test_system.py               # Test suite
├── demo.py                      # Quick demo script
│
└── evaluation_results/          # Generated evaluation outputs
    ├── evaluation_results.csv   # Detailed metrics
    ├── metrics_by_k.png         # Precision/Recall/F1 plots
    └── f1_by_query.png          # Per-query performance
```
