# -*- coding: utf-8 -*-
"""Telecom_Churn_Model_building (3).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nzRpAc03NFJDSf9dbM1dDNJ--gM7NO4q

#Telecom Customer Churn Prediction

## Importing Libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import os

"""# Importing the dataset"""

df = pd.read_excel('/content/churn1.xlsx')
df.head()

# Dropping unnecessary column "Unnamed"
df=df.iloc[:,1:]

# first 5 rows of data
df.head()

# Drop the 'state' column
df = df.drop('state', axis=1)

# information of the dataset
df.info()

""" ### It gives us all columns names, it's data type. Here we can observe that two features daycharge and evening minutes are getting wrong data type. So we will convert it into correct data type."""

df['day.charge']=pd.to_numeric(df['day.charge'],errors='coerce')
df['eve.mins']=pd.to_numeric(df['eve.mins'],errors='coerce')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

columns_to_encode = ['area.code', 'voice.plan', 'intl.plan', 'churn']

# Create mapping dictionaries
mapping_dict = {}

for col in columns_to_encode:
    df[col] = le.fit_transform(df[col])
    # Create a mapping dictionary for the current column
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    mapping_dict[col] = mapping

# Print the mapping dictionaries
mapping_dict

df.isnull().sum()

# median imputation

# we can fill null values by mean but we have outliers so median is best prefered.

df['day.charge']=df['day.charge'].fillna(df['day.charge'].median())
df['eve.mins']=df['eve.mins'].fillna(df['eve.mins'].median())

df.isnull().sum()

df.info()

# Handling imbalance with SMOTE
X = df.drop(columns=['churn'])
y = df['churn']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Recreate DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['churn'] = y_resampled

# Outlier removal using 3-sigma rule, excluding 'voice.plan', 'intl.plan', 'churn'
for col in df_resampled.select_dtypes(include=np.number).columns:
    if col != ('voice.plan', 'intl.plan', 'churn'):  # Exclude 'voice.plan', 'intl.plan', 'churn' column. Even if its not excluded also, no issue. Since boolean columns are not considered.
        mean = df_resampled[col].mean()
        std = df_resampled[col].std()
        df_resampled = df_resampled[(df_resampled[col] > mean - 3 * std) & (df_resampled[col] < mean + 3 * std)]

from sklearn.preprocessing import MinMaxScaler

numerical_features = df_resampled.select_dtypes(include=np.number).columns
scaler = MinMaxScaler()
normalised_data = scaler.fit_transform(df_resampled[numerical_features])
normalised_df = pd.DataFrame(normalised_data, columns=numerical_features, index=df_resampled.index)
normalised_df.head()

X=normalised_df.iloc[:,:-1]
y=normalised_df.iloc[:,-1]

X_rfe = normalised_df.drop(columns=['churn'])
y_rfe = normalised_df['churn']

"""# Collinearity check and feature selection"""

def select_features_by_collinearity(df, threshold=0.85, top_n=10):

    corr_matrix = df.corr()
    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                correlated_features.add(colname)

    # Calculate feature importances (example using Random Forest)
    from sklearn.ensemble import RandomForestClassifier
    X = df.drop(columns=['churn'])  # 'churn' is target variable
    y = df['churn']
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)

    # Filter out correlated features and sort by importance
    filtered_importances = feature_importances.drop(correlated_features)
    top_features = filtered_importances.sort_values(ascending=False).head(top_n).index.tolist()

    return top_features

top_ten_features = select_features_by_collinearity(df_resampled, threshold=0.85, top_n=10)
top_ten_features

"""# Top features based on
1. **mutual information:**
intl.charge   :   0.429700,
intl.mins     :   0.426242,
night.charge  :    0.221482,
day.charge    :    0.144500,
day.mins      :    0.129421,
eve.charge    :    0.100247,
eve.mins      :    0.066443,
night.mins    :    0.054718,
voice.plan    :    0.050531,
voice.messages:    0.032908


2. **non-linear patterns:**
 day.charge     :   0.141592,
 day.mins       :   0.125520,
 customer.calls :   0.095396,
 eve.mins       :   0.061251,
 eve.charge     :   0.059580,
 voice.messages :   0.053899,
 voice.plan     :   0.052052,
 night.charge   :   0.045311,
 intl.charge    :   0.045038,
 night.mins     :   0.043922

3. **RFE:**
 area.code, voice.plan, voice.messages, intl.mins, intl.calls,
       day.mins, day.charge, eve.charge, night.charge,
       customer.calls

4. **Feature selection based on collinearity check.**
  ['day.mins',
 'customer.calls',
 'eve.mins',
 'voice.plan',
 'night.mins',
 'account.length',
 'intl.mins',
 'night.calls',
 'day.calls',
 'eve.calls']
"""

# Splitting into X and y based on collinearty check feature selection, although i can choose mutual info score or nonlinear pattern, i selected rfe for feature selection
X_colinearity_selected = normalised_df[['day.mins',
 'customer.calls',
 'eve.mins',
 'voice.plan',
 'night.mins',
 'account.length',
 'intl.mins',
 'night.calls',
 'day.calls',
 'eve.calls']]
y = normalised_df['churn']

"""X TRAIN AND Y TRAIN SPLITTING"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_colinearity_selected, y, test_size=0.2, random_state=42)



"""# **MODEL BUILDING**

LOGISTIC REGRESSION
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize and train the logistic regression model
logreg = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

"""TUNED XGBOOST MODEL"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Expanded parameter grid
param_grid_xgb = {
    'n_estimators': [150, 200, 250, 300],
    'learning_rate': [0.001, 0.01, 0.1, 0.2],
    'max_depth': [8, 9, 10],
    'subsample': [0.7, 0.8, 0.9, 1.0]
}

# Use RandomizedSearchCV for efficiency
random_search_xgb = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    param_distributions=param_grid_xgb,
    n_iter=50,  # Adjust for more trials
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# Fit the model
random_search_xgb.fit(X_train, y_train)

# Get best parameters
best_params_xgb = random_search_xgb.best_params_
best_score_xgb = random_search_xgb.best_score_

print(f"Best hyperparameters for XGBoost: {best_params_xgb}")
print(f"Best cross-validation score for XGBoost: {best_score_xgb}")

# Train final model with best parameters
best_xgb_classifier = xgb.XGBClassifier(**best_params_xgb, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
best_xgb_classifier.fit(X_train, y_train)

# Evaluate on test data
y_pred_best_xgb = best_xgb_classifier.predict(X_test)
accuracy_best_xgb = accuracy_score(y_test, y_pred_best_xgb)

print(f"XGBoost Accuracy (with best hyperparameters): {accuracy_best_xgb}")
print(classification_report(y_test, y_pred_best_xgb))
print(confusion_matrix(y_test, y_pred_best_xgb))

"""TUNED GRADIENT BOOSTING MODEL"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define hyperparameter grid
param_grid_gb = {
     'n_estimators': [50, 100, 200, 300, 500],  # Number of boosting rounds
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],  # Step size
    'max_depth': [3, 5, 7, 9],  # Tree depth
    'min_samples_split': [2, 5, 10, 15],  # Min samples to split a node
    'min_samples_leaf': [1, 2, 4, 6],  # Min samples per leaf
    'subsample': [0.7, 0.8, 0.9, 1.0],  # Fraction of samples per iteration
    'max_features': ['sqrt', 'log2', None]  # Features per split
}

# Initialize RandomizedSearchCV
random_search_gb = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=param_grid_gb,
    n_iter=50,  # Adjust for more trials
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# Fit the model
random_search_gb.fit(X_train, y_train)

# Get best parameters and score
best_params_gb = random_search_gb.best_params_
best_score_gb = random_search_gb.best_score_

print(f"Best hyperparameters for Gradient Boosting: {best_params_gb}")
print(f"Best cross-validation score: {best_score_gb}")

# Train final model with best parameters
best_gb_classifier = GradientBoostingClassifier(**best_params_gb, random_state=42)
best_gb_classifier.fit(X_train, y_train)

# Evaluate on test data
y_pred_best_gb = best_gb_classifier.predict(X_test)
accuracy_best_gb = accuracy_score(y_test, y_pred_best_gb)

print(f"Gradient Boosting Accuracy (with best hyperparameters): {accuracy_best_gb}")
print(classification_report(y_test, y_pred_best_gb))
print(confusion_matrix(y_test, y_pred_best_gb))

import joblib

# Assuming your trained models are named logistic_model, xgboost_model, and gb_model
joblib.dump(logreg, "logistic_model.pkl")
joblib.dump(best_xgb_classifier, "tuned_xgboost_model.pkl")
joblib.dump(best_gb_classifier, "tuned_gradient_boosting_model.pkl")

"""# DEPLOYMENT"""

!pip install streamlit

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained models
logistic_model = joblib.load("logistic_model.pkl")
xgboost_model = joblib.load("tuned_xgboost_model.pkl")
gb_model = joblib.load("tuned_gradient_boosting_model.pkl")

# Top 10 Features Ranked by Importance
top_features = ['day.mins', 'customer.calls', 'eve.mins', 'voice.plan', 'night.mins',
                'account.length', 'intl.mins', 'night.calls', 'day.calls', 'eve.calls']

# Streamlit UI
st.title("Telecom Churn Prediction App")
st.write("Select model, features, and input values to predict churn.")

# Model selection
st.write("### Tuned Gradient Boosting has the highest accuracy")
model_choice = st.selectbox("Choose a model:",
                            ["Logistic Regression", "Tuned XGBoost", "Tuned Gradient Boosting (Highest Accuracy)"])

# Feature selection
selected_features = st.multiselect("Select features:", top_features, default=top_features)

# User input for selected features
user_input = {}
for feature in selected_features:
    user_input[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

# Convert user input to DataFrame
input_data = pd.DataFrame([user_input])

# Prediction
if st.button("Predict Churn"):
    if model_choice == "Logistic Regression":
        model = logistic_model
    elif model_choice == "Tuned XGBoost":
        model = xgboost_model
    else:
        model = gb_model

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]

    st.subheader("Prediction Result")
    churn_status = "Churned" if prediction[0] == 1 else "Not Churned"
    st.write(f"Prediction: **{churn_status}**")
    st.write(f"Churn Probability: **{probability[0]:.2%}**")

