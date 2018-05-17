# LogisticRegression
Predicting a fraud transaction.

FittingLogisticRegression.ipynb is the code which runs a logistic regression model on a transactions data to predict if the given transaction is fraudulent or not.

## Installation
The code uses Jupyter, hence Jupyter Notebook should be installed.

## Usage
Run the code FittingLogisticRegression.ipynb in Jupyter. The data used is fraud_dataset.csv

## Description

### Data
The fraud_dataset.csv has the columns
**transaction_id**, **duration**, **day**, **fraud**

Fraud is a logical variable which can take the values "True" or "False",
day is a categorical variable which can be either a "weekday" or "weekend".

### Code
Code uses the `logit` function in `statsmodel` package
`log_mod = sm.Logit(df['fraud'], df[['intercept', 'weekday', 'duration']])`

`results = log_mod.fit()`

