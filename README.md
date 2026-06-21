# 🏦 Loan Approval Prediction System

A Machine Learning project that predicts whether a loan application will be **Approved** or **Rejected** based on applicant details such as income, loan amount, CIBIL score, assets, education, and employment status.

This project includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature encoding
- Model training and evaluation
- Model comparison
- Overfitting analysis
- Final model saving using Joblib
- Deployment using Streamlit

---

# 📌 Project Objective

The goal of this project is to build a machine learning model that can predict loan approval status using historical loan application data.

The model helps answer:

> **Will the loan be approved or rejected based on the applicant’s profile?**

---

# 📂 Dataset Features

The dataset contains the following columns:

- `loan_id`
- `no_of_dependents`
- `is_Graduate`
- `self_employed`
- `income_annum`
- `loan_amount`
- `loan_term`
- `cibil_score`
- `residential_assets_value`
- `commercial_assets_value`
- `luxury_assets_value`
- `bank_asset_value`
- `loan_status`

---

# 🎯 Target Variable

- **loan_status**
  - `1` → Approved
  - `0` → Rejected

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed:

## 1. Handling Categorical Columns

The categorical columns were converted into numerical form:

### `is_Graduate`
- Graduate → `1`
- Not Graduate → `0`

### `self_employed`
- Yes → `1`
- No → `0`

### `loan_status`
- Approved → `1`
- Rejected → `0`

---

## 2. Feature Selection

The following columns were used as input features for model training:

```python
[
    'no_of_dependents',
    'is_Graduate',
    'self_employed',
    'income_annum',
    'loan_amount',
    'loan_term',
    'cibil_score',
    'residential_assets_value',
    'commercial_assets_value',
    'luxury_assets_value',
    'bank_asset_value'
]
```

The following columns were **not used as input features**:
- `loan_id`
- `loan_status` (target variable)

---

# 📊 Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset and relationships between features.

## Analysis performed:
- Distribution of loan approval status
- Countplots of categorical features
- Histograms of numerical columns
- Boxplots to detect outliers
- Correlation heatmap
- Relationship between CIBIL score and loan approval
- Income vs Loan Amount analysis
- Asset values vs loan approval

---

# 🤖 Models Applied

The following machine learning models were trained and evaluated:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Decision Tree Classifier
4. Random Forest Classifier
5. K-Nearest Neighbors (KNN)
6. Gradient Boosting Classifier

---

# 📈 Model Performance

| Model | Accuracy |
|------|----------|
| Logistic Regression | 0.9133 |
| SVM | 0.9450 |
| Decision Tree | 0.9813 |
| Random Forest | 0.9801 |
| KNN | 0.8946 |

---

# Overfitting Analysis

During training, **Decision Tree** and **Random Forest** showed signs of overfitting because their training accuracy was **1.0**.

## Example:

### Decision Tree
- **Train Accuracy:** `1.0`
- **Test Accuracy:** `0.9812`

### Random Forest
- **Train Accuracy:** `1.0`
- **Test Accuracy:** `0.9800`

This indicated that the models were memorizing the training data too well.

---

# 🌳 Decision Tree Pruning

To reduce overfitting, a **pruned Decision Tree** was trained.

## Pruned Decision Tree Results
- **Train Accuracy:** `0.9728`
- **Test Accuracy:** `0.9742`

### Classification Report
- Accuracy: **97%**
- Balanced performance across both classes

---

# Final Model: Gradient Boosting Classifier

After comparing all models, **Gradient Boosting Classifier** was selected as the final model because it gave:

- High test accuracy
- Better generalization
- Less overfitting compared to pure Decision Tree / Random Forest
- Strong performance on both approval and rejection classes

## Final Gradient Boosting Results
- **Train Accuracy:** `0.9968`
- **Test Accuracy:** `0.9824`

### Classification Report Summary
- High precision, recall, and F1-score
- Good balance across both classes

---

# Model Saving

The final model and deployment files were saved using **Joblib**.

Saved files:
- `model.pkl` → trained Gradient Boosting model
- `scaler.pkl` → scaler object (`None` for Gradient Boosting if scaling not used)
- `columns.pkl` → list of training input columns

---

#  Streamlit Web App

A Streamlit app was built to allow users to enter loan applicant details and get an instant prediction.

## User Inputs in Streamlit
The app takes the following inputs:

- Number of Dependents
- Education
- Self Employed
- Annual Income
- Loan Amount
- Loan Term
- CIBIL Score
- Residential Assets Value
- Commercial Assets Value
- Luxury Assets Value
- Bank Asset Value

## Output
The app predicts:
- **Loan Approved**
or
- **Loan Rejected**

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

# 📁 Project Structure

```bash
loan_approval_prediction/
│
├── app.py                 # Streamlit application
├── model.pkl              # Trained Gradient Boosting model
├── loan_data.csv          # Dataset
├── notebook.ipynb         # Jupyter notebook with EDA + training
└── README.md              # Project documentation
```

---

#  How to Run the Project

## 1. Clone the repository
```bash
git clone <your-repository-link>
cd loan_approval_prediction
```

## 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

## 3. Run the Streamlit app
```bash
streamlit run app.py
```

---

#  Machine Learning Workflow Followed

1. Load dataset  
2. Clean and preprocess data  
3. Encode categorical features  
4. Perform EDA and visualization  
5. Split data into train/test sets  
6. Train multiple ML models  
7. Compare model performance  
8. Check for overfitting  
9. Apply pruning / boosting techniques  
10. Select final best model  
11. Save model and columns  
12. Build Streamlit app for deployment  

---

# 📌 Conclusion

This project successfully builds a **Loan Approval Prediction System** using machine learning.

### Final Outcome:
- Multiple ML models were tested
- Overfitting was analyzed and reduced
- **Gradient Boosting Classifier** was chosen as the final model
- A **Streamlit app** was created for real-time prediction

This project demonstrates the complete ML pipeline:

> **Preprocessing → EDA → Modeling → Evaluation → Deployment**

---

# ⭐ Future Improvements

Possible future enhancements:
- Add more advanced feature engineering
- Use XGBoost / LightGBM / CatBoost
- Add loan approval probability visualization
- Deploy online using Streamlit Cloud / Render / Hugging Face Spaces
- Add SHAP / feature importance explanation for interpretability
