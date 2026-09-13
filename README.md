# Information Retrieval Programming Assignment II
**Vector Space Model and Ranked Retrieval**

**Group Name:** 200_ok

**Group Members:**

SHANKIT KUMAR DAS - 23CS10066

RITABRATA SARKAR - 23CS30045

SUVADIP BHUIYA - 23CS10070

HRITWIK UPADHYAY - 23CS30023

## Project Description
This project implements a ranked retrieval pipeline using the open-source search engine **PyTerrier**. The pipeline is evaluated on the standard **Cranfield text collection**. 

In accordance with the assignment guidelines, this implementation strictly utilizes sparse vector space models. We evaluate and compare the retrieval performance of:
1. TF-IDF
2. BM25 (Default parameters)
3. BM25 (Tuned parameters via Grid Search)

[GitHub link](https://github.com/suvadipbhuiya2004/information_retrival_assignment_2)

## Directory Structure
Ensure all the following files are in the same directory before running the code:
* `model.py` (The main PyTerrier Python script)
* `cran.all.1400` (The Cranfield document corpus)
* `cran.qry` (The Cranfield test queries)
* `cranqrel` (The human relevance judgments)

## Prerequisites
* Python 3.8+
* Java (OpenJDK 11 is recommended)

## Setup and Execution

1. **Create and activate a virtual environment:**
   ```bash
   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Required Packages:**
   ```bash
   pip install python-terrier pandas scikit-learn
   ```

3. **Run the Pipeline:**
   ```bash
   python model.py
   ```

## Output
Upon successful execution, the script will output:
1. **Indexing Time:** The time taken to parse and index the 1400 documents.
2. **Search Time:** The time taken to execute the test queries.
3. **Comparative Retrieval Performance:** A table showing the MAP, NDCG, and P@10 scores for TF-IDF, BM25 (Default), and BM25 (Tuned).
4. **Result File:** A TSV file named `200_ok_results.txt` containing the ranked retrieval results for the test queries.

## Troubleshooting: macOS ARM64 Java Error
If you are running this on an Apple Silicon Mac (M1/M2/M3) and encounter a `SystemError: Error calling dlopen` or issues with `pyjnius` pointing to an incompatible `x86_64` architecture, you must explicitly set your Java path to an ARM64 JDK before importing PyTerrier.

1. Ensure you have an ARM64 JDK installed (e.g., `brew install openjdk@11`).
2. Add the following to the **very top** of `model.py`, before `import pyterrier`:
   ```python
   import os
   import subprocess
   # Dynamically fetch and set the ARM64 Java path
   java_path = subprocess.check_output(['/usr/libexec/java_home', '-v', '11']).decode('utf-8').strip()
   os.environ["JAVA_HOME"] = java_path
   ```
3. If the error persists, purge the pip cache and reinstall `pyjnius`:
   ```bash
   pip uninstall -y pyjnius
   pip install --no-cache-dir pyjnius
   ```
