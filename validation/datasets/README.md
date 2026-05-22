# Validation Datasets

## Overview

Cleanalytix validation was performed on real-world tabular datasets covering census,
customer churn, credit risk, taxi trips, and credit card behavior.

The raw datasets are excluded from this repository because some files are large
and dataset licenses/distribution rules vary by source. To reproduce validation
locally, download the datasets from the listed sources and place them in the
expected folders below.

## Expected Layout

```text
validation/datasets/
|-- adult/
|   |-- .gitkeep
|   |-- adult.data
|   `-- adult.test
|-- customer_churn/
|   |-- .gitkeep
|   `-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|-- UCI_credit_card/
|   |-- .gitkeep
|   `-- UCI_Credit_Card.csv
|-- yellow_trip_data/
|   |-- .gitkeep
|   |-- yellow_tripdata_2015-01.csv
|   |-- yellow_tripdata_2016-01.csv
|   |-- yellow_tripdata_2016-02.csv
|   `-- yellow_tripdata_2016-03.csv
|-- CC_GENERAL/
|   |-- .gitkeep
|   `-- CC GENERAL.csv
`-- README.md
```

## Dataset List

1. Adult Dataset

Purpose: census-income validation with mixed numeric and categorical columns.

Source: [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult)

Expected files:

```text
validation/datasets/adult/adult.data
validation/datasets/adult/adult.test
```

2. Customer Churn Dataset

Purpose: customer-account validation with categorical, numeric, and identifier columns.

Source: [IBM customer churn sample reference](https://github.com/IBM/customer-churn-prediction)

Expected file:

```text
validation/datasets/customer_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

3. UCI Credit Card Dataset

Purpose: credit-default validation with financial and demographic columns.

Source: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

Expected file:

```text
validation/datasets/UCI_credit_card/UCI_Credit_Card.csv
```

Note: the official UCI release is commonly distributed as `.xls`; this validation
workflow expects a CSV export named `UCI_Credit_Card.csv`.

4. NYC Yellow Taxi Dataset

Purpose: high-volume trip-record validation with numeric, datetime, and location fields.

Source: [NYC TLC trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

Expected files:

```text
validation/datasets/yellow_trip_data/yellow_tripdata_2015-01.csv
validation/datasets/yellow_trip_data/yellow_tripdata_2016-01.csv
validation/datasets/yellow_trip_data/yellow_tripdata_2016-02.csv
validation/datasets/yellow_trip_data/yellow_tripdata_2016-03.csv
```

Note: current TLC files may be distributed as parquet. Convert them to CSV or
provide CSV files with the expected names before running validation.

5. CC GENERAL Dataset

Purpose: credit-card behavior validation with account-level spending metrics.

Source: [Kaggle CC General dataset](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

Expected file:

```text
validation/datasets/CC_GENERAL/CC GENERAL.csv
```

## Setup

Download each dataset manually and place the files in the exact folder paths
shown above. The `.gitkeep` files are placeholders so the empty folder structure
can stay in Git without committing raw datasets.

## Validation

Run the validation script from the repository root:

```bash
python validation/run_validation.py
```

You can also run the notebook:

```text
validation/main.ipynb
```

Validation saves non-empty result tables under:

```text
validation/outputs/<dataset_name>/
```

If a dataset is missing, the validation runner prints a clear message, points to
this README, skips that dataset, and continues with the remaining validations.

## Sampling Note

To keep validation practical while preserving the original workflow, the yellow
taxi validation reads the first `20,000` rows from each configured monthly file.
