<div align="center">

# 🤖 New User Prediction

### Machine Learning System for Predicting New User Conversion

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data--Analysis-150458?style=for-the-badge&logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)

**A machine learning project that analyzes user behavior and predicts whether a new user will convert, register, or perform key engagement actions based on historical data.**

[📊 View Notebooks](#) | [📖 Documentation](#features) | [🐛 Report Bug](https://github.com/TanayV24/New-user-prediction/issues) | [💡 Request Feature](https://github.com/TanayV24/New-user-prediction/issues)

</div>

---

## ✨ Features

### 📊 **Data Science Features**
- 🔍 **Exploratory Data Analysis (EDA)** – Understand user trends & behavior  
- 🧹 **Data Preprocessing** – Handles missing values, encoding, scaling  
- 🧱 **Feature Engineering** – Transformations for better ML performance  
- 🤖 **ML Modeling** – Logistic Regression, Random Forest, SVM, Decision Trees  
- 📈 **Model Evaluation** – Accuracy, precision, recall, F1-score, ROC-AUC  
- 💾 **Model Exporting** – Save trained ML models using joblib/pickle  
- 📊 **Prediction Pipeline** – Predicts new user conversion probability  

### 🔧 **Technical Features**
- 📘 **Jupyter Notebook workflow** – Clean step-by-step structure  
- 🧩 **Modular Python scripts** – preprocessing, models, utils  
- 🚀 **Reproducible ML pipeline**  
- 🔁 **Easy dataset swapping** – Works with any CSV dataset  
- 📂 **Organized codebase** – notebooks + src architecture  

---

## 🛠 Tech Stack

<table>
<tr>
<td width="50%" valign="top">

### Core ML Stack
- **Language:** Python 3.8+  
- **Data Processing:** Pandas, NumPy  
- **ML Models:** Scikit-Learn  
- **Visualization:** Matplotlib, Seaborn  
- **Notebook Environment:** Jupyter Notebook  

</td>
<td width="50%" valign="top">

### Extra Tools
- **Model Saving:** joblib / pickle  
- **Environment:** Virtualenv  
- **File Handling:** CSV datasets  
- **Deployment Ready:** Can integrate with Flask/FastAPI  

</td>
</tr>
</table>

---

## 📋 Prerequisites

Install these tools:

| Tool | Version | Download |
|------|---------|----------|
| 🐍 Python | 3.8+ | https://python.org |
| 📘 Jupyter | Latest | `pip install notebook` |
| 📦 pip | Latest | Comes with Python |
| 💻 Git | Latest | https://git-scm.com |

Check installation:

```

python --version
pip --version
jupyter --version

```

---

## ⚙️ Installation & Setup

### 🚀 Quick Start

1. **Clone the Repository**
```

git clone [https://github.com/TanayV24/New-user-prediction.git](https://github.com/TanayV24/New-user-prediction.git)
cd New-user-prediction

```

2. **Create Virtual Environment**
```

python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

```

3. **Install Dependencies**
```

pip install -r requirements.txt

```

4. **Launch Jupyter Notebook**
```

jupyter notebook

```

---

## 🎮 How to Use

1. Open `/notebooks` folder inside Jupyter  
2. Run notebooks in order:  
   - `0_data_exploration.ipynb`  
   - `1_preprocessing_and_feature_engineering.ipynb`  
   - `2_model_training_and_evaluation.ipynb`  
3. Train ML models and evaluate results  
4. Export final model (joblib/pickle)  
5. Use `src/model.py` to make predictions on new data  

---

## 📁 Project Structure

```

New-user-prediction/
│
├── data/                         # (Optional) dataset folder
│
├── notebooks/                    # Jupyter Notebooks
│   ├── 0_data_exploration.ipynb
│   ├── 1_preprocessing_and_feature_engineering.ipynb
│   └── 2_model_training_and_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py          # Data cleaning functions
│   ├── feature_engineering.py    # Feature transformations
│   ├── model.py                  # Training & prediction utilities
│   └── utils.py                  # Helper functions
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file

````

---

## 🔧 Example Prediction

```python
from joblib import load
import pandas as pd

model = load("model.pkl")

sample = pd.DataFrame([{
    "age": 23,
    "time_on_app": 120,
    "pages_visited": 8,
    "referral": "facebook"
}])

prediction = model.predict(sample)
probability = model.predict_proba(sample)

print("Prediction:", prediction)
print("Confidence:", probability)
````

---

## 🐛 Troubleshooting

<details>
<summary>Jupyter Notebook not opening</summary>

Run:

```
pip install notebook
jupyter notebook
```

</details>

<details>
<summary>Import errors</summary>

Reinstall dependencies:

```
pip install -r requirements.txt --upgrade
```

</details>

<details>
<summary>Model accuracy is low</summary>

Try:

* Scaling data
* Feature selection
* Different ML algorithms
* Hyperparameter tuning

</details>
