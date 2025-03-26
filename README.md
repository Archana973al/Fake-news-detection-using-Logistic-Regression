# ⚠️ Fake-news-detection-using-Logistic-Regression

# Overview
Fake news has become a significant issue in today's digital age. This project aims to detect fake news using machine learning techniques. 

## The notebook walks through the following steps:

### 1. 📥 Data Loading
```python
fake_news = pd.read_csv("fake.csv")
real_news = pd.read_csv("true.csv")
```
### 2.🧹 Data Preprocessing: Cleaning and preparing the text data for analysis.

### 3.🔢 Feature Extraction: Using TF-IDF Vectorization to convert text into numerical features.

### 4.🤖 Model Training: Training a Logistic Regression model on the processed data.

### 5.📊 Evaluation: Evaluating the model's performance using accuracy and a classification report.

## 📂 Dataset

### Composition
| File       | Samples | Content Type          | Source         |
|------------|---------|-----------------------|----------------|
| `Fake.csv` | 23,481  | Fabricated news articles | Kaggle/ISOT    |
| `True.csv` | 21,417  | Verified news reports  | Reuters/BBC    |

### Key Characteristics
```python
print(df.info())
```

## 📦 Dependencies

### Core Libraries
| Library       | Version | Purpose                     |
|---------------|---------|-----------------------------|
| `pandas`      | >=1.3.0 | Data manipulation           |
| `numpy`       | >=1.21.0| Numerical operations        |
| `scikit-learn`| >=1.0   | Machine learning models     |
| `seaborn`     | >=0.11  | Statistical visualizations  |
| `matplotlib`  | >=3.5   | Plotting graphs             |

### Pre-installed Utilities
- `re` (Python built-in): Text pattern matching
- `string` (Python built-in): String operations

### Installation
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```
## Usage
### 1. Clone Repository
```bash
git clone https://github.com/Archana973al/Fake-news-detection-using-Logistic-Regression
cd Fake-news-detection-using-Logistic-Regression
```
### 2.Prepare Data:# Ensure dataset files are in project root:
ls Fake.csv True.csv
### 3.Launch Jupyter:jupyter notebook fakenewsdetection.ipynb
### 4. Execute Notebook
▶️ Run cells sequentially:

Data Loading (Ctrl+Enter)

Preprocessing (Shift+Enter)

Model Training (Shift+Enter)

Evaluation (Shift+Enter)


## Results
![Image](https://github.com/user-attachments/assets/8e62c02a-1578-40fe-9ad2-a8ce47bf449a)

The Logistic Regression model achieves an accuracy of 98.7% on the test dataset. The classification report provides detailed metrics such as precision, recall, and F1-score for both "fake" and "real" news classes.

![Image](https://github.com/user-attachments/assets/53873856-b222-4837-9ec7-9017c85ed20d)

## 📊Confusion Matrix
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
