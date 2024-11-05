# Data Science Project: Credit Card Fraud Detection

![](https://github.com/yogee4/DataScience_Project/blob/main/logo.png)

## What is Credit Card Fraud Detection?

Credit card fraud detection is a critical application of machine learning that uses algorithms to identify and prevent unauthorized transactions. Machine learning models can analyze patterns in transaction data, such as purchase amounts, locations, and times, to detect anomalies that may indicate fraudulent activity.

## Introduction

Credit card fraud detection is crucial for financial institutions to safeguard customers from unauthorized transactions. This project utilizes machine learning models to detect fraudulent transactions using a European cardholder transactions dataset. Given the high imbalance in the dataset, specialized preprocessing and evaluation techniques were employed to ensure the effectiveness of the models.

## Requirements

*Python Libraries:*

- pandas: For data manipulation and analysis.

- numpy: For numerical computations.

- matplotlib: For data visualization.

- sklearn: For implementing machine learning models and preprocessing.

- tensorflow: For building and training neural network models.

## Tools:

Jupyter Notebook or any Python IDE for running the Python code.
A machine with sufficient computational power to train models on a dataset with approximately 284,807 transactions.

## About the Data

*Dataset Description:*

The dataset contains transactions made by European credit cardholders in September 2013.
It consists of 284,807 transactions over two days, with 492 labeled as fraudulent (0.172% of the dataset).

Features:

The dataset has undergone Principal Component Analysis (PCA), resulting in 28 principal components (V1 to V28).

- Time: Seconds elapsed between each transaction and the first transaction in the dataset.

- Amount: Transaction amount.

- Class: The target variable, with 1 indicating fraud and 0 indicating non-fraud.

## Challenges:

Class Imbalance: Only 0.172% of transactions are fraudulent, requiring techniques like balanced sampling, appropriate metrics (e.g., AUPRC), and tailored model evaluation to handle the imbalance.

## Conclusion

The project explores multiple machine learning models for credit card fraud detection, including logistic regression, shallow neural networks, Random Forest, Gradient Boosting, and SVM. The following key points were observed:

*Preprocessing Steps:*

- Data scaling using RobustScaler for the Amount feature.

- Normalization of the Time feature.

- Data shuffling and splitting into training, validation, and test sets.

*Model Performance:*

- Models were evaluated using precision, recall, and F1-score due to the imbalanced nature of the dataset.

- A neural network with appropriate architecture showed promising results on both the imbalanced and balanced datasets.

- Ensemble methods like Random Forest and Gradient Boosting provided robust performance.

- SVM with class-weight adjustment effectively handled the class imbalance.

*Recommendations:*

- Utilizing balanced datasets through under-sampling or over-sampling can improve model performance on minority classes.

- AUPRC is recommended for evaluating model accuracy given the skewed distribution of the target variable.

- By combining preprocessing techniques with a mix of machine learning models, this project demonstrates an effective approach to tackling the challenge of credit card fraud detection.
