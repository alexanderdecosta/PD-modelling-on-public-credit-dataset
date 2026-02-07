# Credit Risk Modeling and Stress Testing

## Project Overview
This notebook demonstrates credit risk modeling on a retail credit dataset (Kaggle “Give Me Some Credit”).  

It covers:  
- Data cleaning and preprocessing  
- Feature engineering tailored for credit risk  
- Lasso Logistic Regression modeling  
- Model evaluation using ROC-AUC, PR-AUC, KS statistic, and Brier score  
- Partial Dependence plots for key variables  
- Stress testing to simulate extreme economic scenarios  

**Goal:** Quantify borrower default risk under both normal and stressed conditions.

---

## Dataset
- **Source:** Kaggle “Give Me Some Credit”  
- **Format:** CSV (`cs-training.csv` for training, `cs-test.csv` for testing)  
- Columns renamed for readability (e.g., `NumberOfTimes90DaysLate` → `Num90Plus_DPD`)  
- **Note:** Data is not included in this repo. Users should download from Kaggle: [https://www.kaggle.com/competitions/GiveMeSomeCredit/data](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)

---

## Feature Engineering
Key transformations applied to improve model interpretability and performance:  

- **Skewed variables (`MonthlyIncome`, `DebtRatio`):** log1p transformation to reduce skew  
- **Missing values:** Simple imputer replaces NaN with 0 (works well for log regression as then that term has no effect on `logit(p)`)
- **Numeric variables:** Scaled with `RobustScaler` to standardize feature scales for Lasso  
- **Revolving Utilization (`RevUtil`):** Split into three variables:  
  - `RevUtil_zero`: 1 if near-zero utilization (reduces PD)  
  - `RevUtil_one`: 1 if maxed out on credit (increases PD)  
  - `RevUtil_mid`: intermediate values between 0 and 1  

This captures non-linear effects in a logistic regression framework.

---

## Model Details
- **Model:** Lasso Logistic Regression (`penalty='l1'`)  
- **Solver:** `liblinear`, `C=0.01`, `max_iter=1000`  

**Validation Performance:**  
- ROC AUC: 0.849  
- PR AUC: 0.381  
- Optimal threshold: 0.160  
- F1-score at optimal threshold: 0.430  
- KS Statistic: 0.411  
- Brier Score: 0.0505  

**Coefficients:**

| Feature             | Coefficient |
|--------------------|------------|
| RevUtil_mid         | 1.206      |
| RevUtil_zero        | -0.414     |
| RevUtil_one         | 1.491      |
| DebtRatio           | 0.113      |
| MonthlyIncome       | -0.040     |
| age                 | -0.383     |
| Num30_59_DPD        | 0.439      |
| NumOpenCredit       | 0.096      |
| Num90Plus_DPD       | 0.619      |
| NumRealEstateLoans  | 0.086      |
| Num60_89_DPD        | 0.565      |
| NumDependents       | 0.013      |
| DebtRatio_missing_flag | -0.296 |

**Insight:** Delinquency variables (`Num30_59_DPD`, `Num60_89_DPD`, `Num90Plus_DPD`) and `RevUtil` dominate model predictions. `MonthlyIncome` has surprisingly low impact.

---

## Stress Testing
Simulate economic stress scenarios and examine predicted PDs:  

| Scenario                | Description           | Key Feature Changes |
|------------------------|---------------------|-------------------|
| Moderate Downturn       | Mild economic stress | `MonthlyIncome ×0.8`, `DebtRatio ×1.5`, `delinquency variables +1` |
| Severe Downturn         | Major economic stress | `MonthlyIncome ×0.5`, `DebtRatio ×2.0`, `delinquency variables +2/+1` |
| Revolving Credit Spike  | Borrowers use more credit | `RevUtil ×1.5` |

**Expected PDs per scenario:**

| Scenario             | Mean PD | Median PD | 90th percentile |
|--------------------|---------|-----------|----------------|
| Base Case           | 0.0831  | 0.0430    | 0.1777         |
| Moderate Downturn   | 0.2512  | 0.1899    | 0.5307         |
| Severe Downturn     | 0.4330  | 0.3979    | 0.7610         |
| High RevUtil        | 0.0961  | 0.0493    | 0.2140         |

**Insight:** Stress scenarios increase expected PD substantially, especially under extreme debt or delinquency conditions.

---

## Notebook Structure
1. Load and clean data  
2. Feature engineering  
3. Train/validation split  
4. Build Lasso Logistic Regression pipeline  
5. Evaluate model performance  
6. Partial Dependence plots  
7. Stress testing and expected PD computation  

---

## Author
Alexander De Costa
