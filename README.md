# Task 1: Data Preprocessing for Machine Learning

Internship task for Codveda Technologies. I prepared a raw customer churn dataset for machine learning using Python, pandas and scikit-learn.

## What I did
1. Inspected the data (size, column types, missing values, duplicates, outliers)
2. Removed duplicate rows and dropped the customer_id column
3. Split the data into 80% training and 20% testing sets before any fitting, to avoid data leakage
4. Filled missing values (median for numbers, most frequent value for text)
5. Encoded categories (ordinal encoding for plan, one-hot encoding for city and gender)
6. Scaled age, income and tenure_months with StandardScaler

## Files
- `generate_data.py`: creates the synthetic raw dataset
- `raw_customers.csv`: the raw data, with missing values, duplicates and outliers
- `data_preprocessing.ipynb`: the full preprocessing notebook
- `train_processed.csv` and `test_processed.csv`: the final processed data

## Tools
Python, pandas, NumPy, scikit-learn

## How to run
pip install pandas numpy scikit-learn
python generate_data.py
Then open the notebook and run all cells.
