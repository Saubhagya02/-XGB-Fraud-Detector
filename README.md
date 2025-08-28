# XGB Fraud Detector | Fraud Detection Model using XGBoost
# Financial Fraud Detection System

## 📜 Project Overview

This project develops a robust machine learning model for a financial company to proactively detect fraudulent transactions. The model analyzes a large-scale transactional dataset to identify patterns indicative of fraud, focusing on transactions where agents attempt to take control of customer accounts and empty the funds.

Beyond a baseline model, this project incorporates advanced validation, tuning, and explainability techniques to create a high-performing, stable, and transparent solution.

The dataset consists of 6,362,620 transactions from a 30-day simulation, providing a rich environment for building and testing a fraud detection system.

---

## 📋 Executive Summary & Task Answers

This section directly addresses the key questions outlined in the project task, reflecting the final, tuned model.

### 1. Data Cleaning
The data was cleaned by:
* **Missing Values**: Confirming no missing values were present.
* **Outliers**: Analyzing outliers, which were kept as they are significant indicators of fraud.
* **Multi-collinearity**: Assessing correlations, which were deemed acceptable for the robust XGBoost model.

### 2. Fraud Detection Model Description
The final model is a **tuned XGBoost (Extreme Gradient Boosting) Classifier**. To handle the severe class imbalance, **SMOTE** was applied to the training data. The model's hyperparameters were then optimized using **GridSearchCV** to maximize predictive performance.

### 3. Variable Selection
Variables were selected through:
* **Domain Knowledge**: Dropping identifier columns (`nameOrig`, `nameDest`) to prevent overfitting.
* **Feature Engineering**: Creating powerful features like `balance_error_orig` and `emptied_account` to capture suspicious activity.
* **Data Transformation**: One-hot encoding the categorical `type` feature.

### 4. Model Performance Demonstration
Model performance was validated and demonstrated using a rigorous, multi-step process:
* **Robustness Check**: **Stratified K-Fold Cross-Validation** was used to confirm the baseline model's stability and high performance across different data subsets.
* **Final Evaluation**: Model performance was validated and demonstrated using a rigorous, multi-step process. The final tuned model achieved an **AUC-ROC Score of 0.9997** on the test data. The detailed classification report showed a **Fraud Recall of 1.00** and a **Fraud Precision of 0.87**, proving the model's high effectiveness in identifying fraud while minimizing false positives.

### 5. Key Factors Predicting Fraud
The key predictors of fraud, confirmed by the model's feature importance and **SHAP analysis**, are:
1.  **Balance Discrepancies**: Errors in the post-transaction balance calculation.
2.  **Account Depletion**: Transactions that drain the originator's account.
3.  **Transaction Type**: Specifically, `TRANSFER` transactions.

### 6. Sense-Checking the Predictive Factors
**Yes, these factors make complete logical sense.** They directly align with the fraudulent behavior of "emptying the funds" via "transferring to another account"[cite: 8]. The model's reliance on these factors proves it has learned the true underlying patterns of fraud in this dataset.

### 7. Recommended Prevention Strategies
A multi-layered prevention strategy is recommended:
* **Real-Time Model Scoring**: Deploy the tuned XGBoost model to score transactions in real-time.
* **Tiered Response System**: Automatically block high-risk transactions and flag medium-risk ones for manual review.
* **Enhanced Dynamic Rules**: Use model insights to create smarter rules (e.g., flag any transaction that empties an account).

### 8. Measuring Implementation Success
Effectiveness should be measured via:
* **KPI Monitoring**: Tracking the **Fraud Detection Rate** (Recall) and **False Positive Rate**.
* **A/B Testing**: Using a Champion/Challenger framework to scientifically compare the new system against the old one.

---

🚀 Final Model Performance
The model refinement process yielded significant improvements over the baseline model. By tuning the hyperparameters, we created a more precise and reliable fraud detection system.
The table below compares the performance of the initial default XGBoost model against the final tuned model on the unseen test data.

### Performance Comparison: Baseline vs. Tuned Model

| Metric            | Baseline (Default) Model | Final (Tuned) Model | Improvement |
| :---------------- | :----------------------- | :------------------ | :---------- |
| **AUC-ROC Score** | 0.9994                   | **0.9997** | ▲ Improved  |
| **Fraud Recall** | **1.00** | **1.00** | ▬ Maintained |
| **Fraud Precision** | 0.87                     | **0.90** | ▲ Improved  |
| **Fraud F1-Score** | 0.93                     | **0.95** | ▲ Improved  |

### Interpretation:

* A **Recall of 1.00** is outstanding, as it means the model successfully identified **100%** of the actual fraudulent transactions in the test set.
* A **Precision of 0.87** indicates that when the model flags a transaction as fraudulent, it is correct 87% of the time, keeping false alarms relatively low.

---

## 🏆 Advanced Techniques & Model Refinement

To ensure the model is not just accurate but also robust and trustworthy, the following advanced techniques were implemented.

### 1. Robust Validation with Stratified K-Fold Cross-Validation
* **What**: The baseline XGBoost model was validated using a 5-fold stratified cross-validation process.
* **Why**: A single train-test split can be subject to luck. Cross-validation provides a more reliable estimate of the model's performance by training and testing it on 5 different subsets of the data, ensuring its high score is consistent and generalizable.

### 2. Hyperparameter Tuning with GridSearchCV
* **What**: The model's internal parameters (hyperparameters) were fine-tuned using `GridSearchCV`. To manage the large dataset (over 10 million rows after SMOTE), this tuning was performed on a stratified sample.
* **Why**: The default parameters of a model are not always optimal. Tuning parameters like `max_depth` and `learning_rate` squeezes the best possible performance out of the model, often leading to better fraud capture. Using a sample is a professional best practice to make this process computationally feasible.

### 3. Model Explainability with SHAP
* **What**: The SHAP (SHapley Additive exPlanations) library was used to interpret the final model's predictions.
* **Why**: For a critical application like fraud detection, it’s not enough to know *that* a transaction is flagged; we need to know *why*. SHAP provides transaction-specific explanations, showing exactly which features contributed to a fraud prediction. This builds trust and provides invaluable insights for human fraud analysts.

---

## 💾 Dataset

The data is a synthetic dataset generated using PaySim, which mimics real-world financial transactions.
* Size: 6,362,620 rows and 10 columns.
* Timeframe: The simulation covers a period of 30 days, where each `step` represents one hour.

---

## 🛠️ How to Use This Repository
### Prerequisites
* Python 3.8+
* pip package manager

### Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  Install the required libraries from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Notebook
Open and run the cells in the `Model.ipynb` file using Jupyter Lab or VS Code.

---

## 📚 Libraries Used
* pandas
* numpy
* scikit-learn
* imbalanced-learn
* xgboost
* matplotlib
* seaborn

---

To create the `requirements.txt` file, run this command in your activated virtual environment:
`pip freeze > requirements.txt`
