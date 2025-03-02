# breast-cancer-classification
# Breast Cancer Classification Model

## Overview
This project builds a **Machine Learning model** to classify breast cancer tumors as **Malignant (0)** or **Benign (1)** using the **Breast Cancer Wisconsin Dataset** from `sklearn.datasets`.

## Dataset Information
- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Features:** 30 numerical attributes related to tumor characteristics
- **Target:**
  - `0` → Malignant (Cancerous)
  - `1` → Benign (Non-Cancerous)
- **Class Distribution:**
  - Benign: 357 samples
  - Malignant: 212 samples

## Project Steps
### 1️⃣ Load the Dataset
```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
```

### 2️⃣ Exploratory Data Analysis (EDA)
- Check for missing values, data types, and class distribution.
```python
print(df.info())
print(df.describe())
print(df['target'].value_counts())
```
- **Plot class distribution**:
```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x=df['target'])
plt.title("Class Distribution (0 = Malignant, 1 = Benign)")
plt.show()
```

### 3️⃣ Data Preprocessing
- **Train-Test Split** (80%-20%)
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['target'])  # Features
y = df['target']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
- **Feature Scaling** (Standardization)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 4️⃣ Model Training (Logistic Regression)
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

### 5️⃣ Model Evaluation
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))
```

### 6️⃣ Confusion Matrix Visualization
```python
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix for Breast Cancer Classification")
plt.show()
```

## Results
- Achieved an **accuracy of ~95%**.
- The confusion matrix shows the correct and incorrect classifications.

## Dependencies
Install required libraries using:
```sh
pip install numpy pandas scikit-learn seaborn matplotlib
```

## Future Improvements
- Try **Random Forest, SVM, or Neural Networks** for better accuracy.
- Implement **hyperparameter tuning** for improved performance.
- Deploy the model using **Flask or Streamlit**.

---


