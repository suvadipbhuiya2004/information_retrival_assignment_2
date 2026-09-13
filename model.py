import pyterrier as pt
import pandas as pd
import time
import re

# 1. Initialization (Updated to clear deprecation warning)
if not pt.java.started():
    pt.init()

# 2. Local Data Parsers
def load_corpus(filepath):
    """Parses cran.all.1400 into a dictionary iterator for indexing."""
    with open(filepath, 'r') as f:
        content = f.read()
    for doc in content.split('.I ')[1:]:
        lines = doc.strip().split('\n')
        doc_id = lines[0].strip()
        # Keep all text lines but strip the structural tags (.T, .A, .W, .B)
        text_lines = [line for line in lines[1:] if not (line.startswith('.') and len(line) <= 3)]
        yield {'docno': doc_id, 'text': " ".join(text_lines).strip()}

def load_queries(filepath):
    """Parses cran.qry into a DataFrame and strips punctuation."""
    queries = []
    with open(filepath, 'r') as f:
        content = f.read()
    for q in content.split('.I ')[1:]:
        lines = q.strip().split('\n')
        qid = lines[0].strip()
        text_lines = [line for line in lines[1:] if not (line.startswith('.') and len(line) <= 3)]
        # Remove punctuation to prevent PyTerrier query parsing errors
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', " ".join(text_lines))
        queries.append({'qid': qid, 'query': clean_text.strip()})
    return pd.DataFrame(queries)

def load_qrels(filepath):
    """Parses cranqrel into a DataFrame and formats relevancy scores."""
    qrels = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                # Invert Cleverdon codes: 1 (best) -> 4, 4 (worst) -> 1
                relevance = 5 - int(parts[2])
                qrels.append({'qid': parts[0], 'docno': parts[1], 'label': relevance})
    return pd.DataFrame(qrels)

# Load the provided local files
topics = load_queries("cran/cran.qry")
qrels = load_qrels("cran/cranqrel")

# 3. Preprocessing & Indexing
start_index_time = time.time()
indexer = pt.IterDictIndexer("./cranfield_index", overwrite=True, blocks=True)
indexref = indexer.index(load_corpus("cran/cran.all.1400"))
indexing_time = time.time() - start_index_time
print(f"Indexing Time: {indexing_time:.2f} seconds")

# 4. Define Sparse Retrieval Models
tfidf = pt.BatchRetrieve(indexref, wmodel="TF_IDF")
bm25_default = pt.BatchRetrieve(indexref, wmodel="BM25")

# 5. Parameter and Model Tuning
print("Tuning BM25 parameters...")
bm25_tune = pt.BatchRetrieve(indexref, wmodel="BM25", controls={"bm25.b": 0.75, "bm25.k_1": 1.2})
param_grid = {
    bm25_tune: {
        "bm25.b": [0.3, 0.5, 0.75, 0.9],
        "bm25.k_1": [0.5, 1.2, 1.5, 2.0]
    }
}

# Split queries for tuning/testing
train_topics = topics.head(100)
test_topics = topics.tail(125)

tuned_bm25 = pt.GridSearch(
    bm25_tune,
    param_grid,
    train_topics,
    qrels,
    "map"
)

# 6. Evaluation
print("Evaluating models...")
start_search_time = time.time()
experiment_results = pt.Experiment(
    [tfidf, bm25_default, tuned_bm25],
    test_topics,
    qrels,
    eval_metrics=["map", "ndcg", "P_10"],
    names=["TF-IDF", "BM25 (Default)", "BM25 (Tuned)"]
)
search_time = time.time() - start_search_time

print(f"Search Time: {search_time:.2f} seconds")
print("\nComparative Retrieval Performance:")
print(experiment_results)

# 7. Output Generation
group_prefix = "200_ok"
output_filename = f"{group_prefix}_results.txt"
final_results = tuned_bm25.transform(test_topics)
final_results.to_csv(output_filename, sep='\t', index=False)
print(f"Result file for test queries saved to {output_filename}")