# Fraud Detection and Risk Project

This project demonstrates a complete workflow for detecting and managing multiple types of fraud risk within a synthetic transaction dataset.  It includes:

* A **synthetic dataset** generator that creates a realistic collection of financial transactions with multiple correlated target variables.
* An **exploratory data analysis** (EDA) script to understand the data distribution and relationships between features and targets.
* A **model training** script that fits machine‑learning models to predict several risk indicators at once and logs experiments using **MLflow**.
* A simple project structure that can be uploaded to GitHub or used as a starting point for more advanced fraud‑analytics work.

## Project structure

```
fraud_detection_project/
├── .github/workflows
│   ├── cd.yml                    # automated containerization, infrastructure and deployment
│   └── ci.yml                    # automated unit and integration testing with linting and formatting checks
├── .venv
├── Dockerfile                    # everything you need for image creation
├── Makefile                      # simplify your command line
├── README.md                     # project overview and instructions
├── app
│   ├── __pycache__
│   └── main.py                   # FastAPI webapp  
├── data
│   ├── generate_data.py          # script to create the synthetic dataset
│   └── transactions.csv          # generated dataset (not committed by default)
├── lambda
│   ├── __pycache__
│   └── handler.py
├── lambda_function_payload.zip
├── mlflow.db
├── mlruns
├── monitoring
│   ├── __pycache__
│   └── monitor.py                # data and concept drift monitoring
├── notebooks
│   ├── eda.ipynb                 # exploratory data analysis script
│   └── unsupervised.ipynb        # dimensionality reduction techniques
├── orchestration
│   ├── __pycache__
│   └── flow.py                   # ml orchestration with prefect
├── plots                         # directory with plot output for eda and dimensionality reduction
├── pyproject.toml                # uv dependencies
├── requirements.txt    
├── src
│   ├── __pycache__
│   └── train.py                  # automated training file for fruad detection using XGboost and ensemble methods
├── terraform                     # infrastructure as code
├── tests                         # unit and integration tests
│   ├── __pycache__
│   ├── test_app.py
│   ├── test_integration.py
│   └── test_train.py
└── uv.lock
```

## Dataset description

Realistic, publicly available datasets that simultaneously measure fraud likelihood, chargeback risk, account takeover probability, and anomaly score are difficult to find due to privacy concerns.  To illustrate multivariate fraud‑analysis techniques, we therefore generate a **synthetic dataset** with the following characteristics:

* **Features:** 20 numerical variables labelled `feature_0` … `feature_19` that simulate transaction attributes (amount, balance history, customer metadata, etc.).  Some features carry stronger signal for fraud and other risks, while others are redundant or noisy.
* **Targets:**
  * `fraud_label` – a binary indicator (0/1) describing whether the transaction is fraudulent.
  * `chargeback_label` – a binary indicator representing the risk that the transaction will result in a chargeback.  It is correlated with, but not identical to, the fraud label.
  * `takeover_label` – a binary indicator flagging the probability of an account takeover event.
  * `anomaly_score` – a continuous value between 0 and 1 representing the degree to which a transaction is unusual.  This score is derived from an unsupervised anomaly‑detection algorithm.

The dataset is created using `data/generate_data.py`.  The script leverages scikit‑learn utilities (`make_classification`, `IsolationForest`) and basic random sampling to produce correlated risk outcomes.  Running the script writes a CSV file (`transactions.csv`) containing 10 000 synthetic transactions.

## Getting started

1. **Set up a Python environment.**  From the project root, create a virtual environment and install dependencies.  We recommend using [uv](https://github.com/astral-sh/uv) for faster dependency installation.  If `uv` is not installed, install it first with `pip install uv`.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   # Install dependencies using uv (a drop‑in replacement for pip).  If you
   # haven't installed uv yet, run `pip install uv` first.
   uv pip install -r requirements.txt
   ```

2. **Generate the dataset.**  Create the `data/transactions.csv` file with 10 000 rows (you can adjust `--n_samples`):

   ```bash
   python data/generate_data.py --n_samples 10000 --output data/transactions.csv
   ```

3. **Explore the data.**  Run the EDA script to view basic statistics and plots (figures will be saved to the `plots/` directory):

   ```bash
   python src/eda.py --data data/transactions.csv
   ```

4. **Train the models.**  Fit predictive models for each risk indicator and log the experiments to an MLflow tracking server (running locally by default).  MLflow runs will be stored under the `mlruns` directory:

   ```bash
   python src/train.py --data data/transactions.csv
   ```

5. **View the MLflow UI (optional).**  To inspect experiment metrics and compare models, launch the MLflow tracking UI:

   ```bash
   mlflow ui
   ```
   Then visit `http://127.0.0.1:5000` in your browser.

6. **Explore unsupervised techniques (optional but great for interviews).**  Unsupervised learning reveals hidden structure in the data and makes for compelling visualisations.  Run the unsupervised script to produce two‑dimensional embeddings of the transactions using PCA, MDS, Isomap, UMAP, t‑SNE, and Isolation Forest:

   ```bash
   python src/unsupervised.py --data data/transactions.csv --sample_size 3000
   ```

   The script will create a `plots/unsupervised/` directory containing a scatter plot for each technique.  These visuals can enrich your portfolio and illustrate your ability to apply manifold learning and anomaly detection.

## Notes

* The dataset generator and scripts are self‑contained—no external accounts or proprietary data are required.
* The project uses only open‑source libraries (`pandas`, `numpy`, `scikit‑learn`, `matplotlib`, `mlflow`) that can be installed from PyPI.
* All plots are created with `matplotlib` without specifying colors, complying with typical guidelines for reproducibility.
* Feel free to extend this skeleton by adding feature engineering, hyper‑parameter tuning, additional models, or a notebook for interactive analysis.
