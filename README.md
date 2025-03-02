
# **Exploring the Link Between Pregnancy Frequency and Diabetes Risk**

## **Project Overview**
This project explores the relationship between the **number of pregnancies** and the **risk of developing diabetes**. Using statistical analysis and machine learning techniques, the study tests the hypothesis that **an increased number of pregnancies correlates with a higher risk of diabetes**. The dataset used is the **Pima Indians Diabetes Database**, sourced from Kaggle.

## **Motivation**
Diabetes is a chronic medical condition that affects millions of people worldwide. **Gestational diabetes**, which occurs during pregnancy, can increase the risk of **developing type 2 diabetes** later in life. By analyzing data on pregnancy frequency and other health metrics, this project provides insights that may aid in **early detection and prevention strategies**.

## **Hypothesis**
- **H₀ (Null Hypothesis):** The mean number of pregnancies for diabetic and non-diabetic women is the same.
- **H₁ (Alternative Hypothesis):** Women with more pregnancies have a significantly higher risk of diabetes.

## **Dataset**
The primary dataset used is the **Pima Indians Diabetes Database**:
- **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Originally from:** **National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)**
- **Features:**
  - **Independent variables:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
  - **Dependent variable (Outcome):** **1** (diabetic) or **0** (non-diabetic)

For this analysis, the focus is on **Pregnancies, Glucose, Blood Pressure, BMI, and Age**.

## **Methodology**
1. **Data Preprocessing**:
   - Loaded the dataset and removed duplicates.
   - Replaced missing values (zeros) in **Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI** with appropriate statistical values.
   - Visualized data distributions using **histograms** and **descriptive statistics**.

2. **Exploratory Data Analysis (EDA)**:
   - Analyzed histograms and distributions for the key variables.
   - Used **Probability Mass Function (PMF)** and **Cumulative Distribution Function (CDF)** to compare pregnancy counts for diabetic vs. non-diabetic women.
   - Conducted **scatter plot** and **correlation analysis** between pregnancies, glucose, and BMI.

3. **Hypothesis Testing**:
   - Performed an **independent t-test** to compare pregnancy counts between diabetic and non-diabetic women.
   - Evaluated statistical significance using **p-values**.

4. **Predictive Modeling**:
   - Implemented **logistic regression** to assess how pregnancies and other factors contribute to diabetes risk.
   - Evaluated model performance using **coefficients and statistical significance**.

## **Key Findings**
- **Pregnancies**: Women with **higher pregnancy counts** were more likely to be diabetic.
- **Glucose & BMI**: Strongly associated with diabetes risk.
- **T-test Results**: The **p-value (6.82e-09)** indicates a statistically significant difference in pregnancy counts between diabetic and non-diabetic women.
- **Regression Results**:
  - **Pregnancies (Coefficient: 0.1429, p < 0.001)** → More pregnancies increase diabetes risk.
  - **Glucose (Coefficient: 0.0377, p < 0.001)** → Higher glucose levels significantly raise diabetes risk.
  - **BMI (Coefficient: 0.0951, p < 0.001)** → Higher BMI is linked to higher diabetes risk.
  - **Blood Pressure (p = 0.432)** → Not a significant predictor in this model.

## **Required Packages**
To run this project, install the following **Python** libraries:
```python
pip install pandas numpy seaborn matplotlib scipy statsmodels thinkstats2 thinkplot
```

**Libraries used**:
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import thinkstats2
import thinkplot
import statsmodels.api as sm
from scipy.stats import ttest_ind
```

## **Running the Project**
1. **Load dataset**:
   ```python
   df = pd.read_csv('diabetes.csv')
   df_clean = df.drop_duplicates()
   ```
2. **Preprocess data**:
   ```python
   columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
   for col in columns_with_zeros:
       mean_value = df_clean[df_clean[col] != 0][col].mean()
       df_clean[col] = df_clean[col].replace(0, mean_value)
   ```
3. **Visualize data**:
   ```python
   sns.histplot(df_clean['Pregnancies'], bins=10, kde=True)
   plt.title('Histogram of Pregnancies')
   plt.show()
   ```
4. **Perform hypothesis testing**:
   ```python
   t_stat, p_value = ttest_ind(df_clean[df_clean['Outcome'] == 1]['Pregnancies'],
                               df_clean[df_clean['Outcome'] == 0]['Pregnancies'], equal_var=False)
   print(f'p-value: {p_value}')
   ```
5. **Train logistic regression model**:
   ```python
   X = df_clean[['Pregnancies', 'Glucose', 'BloodPressure', 'BMI']]
   y = df_clean['Outcome']
   X = sm.add_constant(X)
   model = sm.Logit(y, X).fit()
   print(model.summary())
   ```

## **Implications**
- **Healthcare professionals** can use these findings for **early diabetes screening** in women with multiple pregnancies.
- **Public health initiatives** could target pregnant women for **lifestyle interventions** to reduce diabetes risk.

## **Limitations & Future Improvements**
- **Dataset limitations**: The data is specific to a group and may not generalize globally.
- **Additional features**: Including **Insulin levels, Age, and Diabetes Pedigree Function** could improve prediction accuracy.
- **Model enhancement**: Trying **Random Forest** or **Support Vector Machines (SVM)** could improve accuracy.

## **Conclusion**
This project provides strong statistical evidence that **women with more pregnancies have a higher risk of developing diabetes**. The findings highlight the importance of monitoring glucose levels and BMI in women with multiple pregnancies, reinforcing the role of **data science in healthcare**.

## **References**
- [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [American Diabetes Association](https://www.diabetes.org/diabetes)

---
