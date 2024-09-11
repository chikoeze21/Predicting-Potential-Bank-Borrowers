## Project: Predicting Potential Borrowers for Bank XYZ

### Business Objective
Bank XYZ is focused on growing its borrower base to drive higher loan interest revenue. Currently, the bank has a majority of liability customers (depositors) compared to asset customers (borrowers). A recent marketing campaign yielded a single-digit conversion rate, and the goal is to double that figure with better-targeted marketing campaigns, using the same budget. 

As a data scientist, you are tasked with developing a machine learning model to identify potential borrowers from the existing liability customer base. This will help the bank run more focused marketing campaigns, increasing the chances of converting depositors into borrowers.

### Data Description
The project uses two datasets:
- **Data1**: 5000 rows and 8 columns
- **Data2**: 5000 rows and 7 columns

### Aim
The goal is to build a machine learning model that predicts which liability customers are likely to convert into asset customers. This will help optimize digital marketing efforts.

### Tech Stack
- **Language**: Python
- **Libraries**: 
  - Data Manipulation: `numpy`, `pandas`
  - Data Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `sklearn`, `imblearn`
  - Model Saving: `pickle`

### Approach

1. **Data Preprocessing**
   - Importing the necessary libraries and loading the dataset.
   - Merging the two datasets to create a unified data structure.
   - Handling missing values and dropping irrelevant columns.
   - Checking for multicollinearity and removing highly correlated features.

2. **Exploratory Data Analysis (EDA)**
   - Performing data visualization to understand key trends and relationships.
   - Conducting feature engineering to enhance the predictive power of the dataset.

3. **Model Building**
   - Splitting the data into training and testing sets.
   - Training various machine learning models:
     - Logistic Regression
     - Weighted Logistic Regression
     - Naive Bayes
     - Support Vector Machine (SVM)
     - Decision Tree Classifier
     - Random Forest Classifier

4. **Model Validation**
   - Evaluating model performance using:
     - Accuracy score
     - Confusion matrix
     - Area Under Curve (AUC)
     - Recall score
     - Precision score
     - F1-score
   - Addressing class imbalance with `imblearn` (oversampling/undersampling techniques).
   
5. **Hyperparameter Tuning**
   - Tuning model parameters using `GridSearchCV` to optimize the SVM model's performance.

6. **Model Saving**
   - Saving the best-performing model as a pickle file for later use.

### Conclusion
This project delivers a machine learning model that supports Bank XYZ’s marketing team in identifying potential borrowers, allowing for better-targeted campaigns and a higher conversion rate, contributing to the bank’s revenue growth.

