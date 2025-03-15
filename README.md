# Fake-news-detection-using-Logistic-Regression

# Overview
Fake news has become a significant issue in today's digital age. This project aims to detect fake news using machine learning techniques. The notebook walks through the following steps:

Data Loading: Loading fake and real news datasets.

Data Preprocessing: Cleaning and preparing the text data for analysis.

Feature Extraction: Using TF-IDF Vectorization to convert text into numerical features.

Model Training: Training a Logistic Regression model on the processed data.

Evaluation: Evaluating the model's performance using accuracy and a classification report.

# Dataset
The dataset used in this project consists of two CSV files:

Fake.csv: Contains fake news articles.

True.csv: Contains real news articles.

# Dependencies
To run the code in this notebook, you need the following Python libraries:

pandas

numpy

scikit-learn

seaborn

matplotlib

re

string


You can install the required libraries using the following command:

pip install pandas numpy scikit-learn seaborn matplotlib

# Usage
1.Clone this repository to your local machine:

git clone https://github.com/Archana973al/Fake-news-detection-using-Logistic-Regression

2.Ensure the dataset files (Fake.csv and True.csv) are in the same directory as the notebook.

3.Open the Jupyter notebook:
jupyter notebook fakenewsdetection.ipynb

4.Run the notebook cells sequentially to load the data, preprocess it, train the model, and evaluate its performance.

# Results
![Image](https://github.com/user-attachments/assets/8e62c02a-1578-40fe-9ad2-a8ce47bf449a)

The Logistic Regression model achieves an accuracy of 98.7% on the test dataset. The classification report provides detailed metrics such as precision, recall, and F1-score for both "fake" and "real" news classes.

![Image](https://github.com/user-attachments/assets/53873856-b222-4837-9ec7-9017c85ed20d)

Confusion Matrix
The confusion matrix shows the number of correct and incorrect predictions:

True Positives (TP): Correctly predicted fake news.

True Negatives (TN): Correctly predicted real news.

False Positives (FP): Real news incorrectly predicted as fake.

False Negatives (FN): Fake news incorrectly predicted as real.

Confusing matrix:![Image](https://github.com/user-attachments/assets/30c71598-edee-4248-9e41-5904f88df304)


manual input:

![Image](https://github.com/user-attachments/assets/d197e1f1-b5b4-4ade-9629-6c0f20eb757c)

# Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

Fork the repository.

Create a new branch for your feature or bugfix.

Commit your changes.

Submit a pull request.

# License
This project is licensed under the MIT License. See the LICENSE file for details.
