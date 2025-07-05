# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:33:10 2024

@author: ELLI0T
"""

#Data Cleaning of the dataset and summary statistics
import pandas as pd
import os
import numpy as np
import math
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
import plotly

# READING THE DATASET INTO A DATAFRAME
cvd = pd.read_csv("D:\Semester 3\AIT 664\Research Project\heart.csv")

# EDA

# ------------------------------------------------------CLEANING----------------------------------------------
cvd.info()
cvdscrb = cvd.describe()
cvd.nunique()
cvd.isna().count()
cvd.head()
# Dropping id column from the dataset
cvd = cvd.drop(['id'], axis=1)
# Converting the age from days to years
cvd['age'] = (cvd['age']/365).astype('int')
# Removing outliers for ap_hi and ap_lo
cvd = cvd[(cvd['ap_lo'] < cvd['ap_hi'])]
cvd = cvd[(cvd['ap_hi'] >= 60) & (cvd['ap_hi'] <= 250)]
cvd = cvd[(cvd['ap_lo'] >= 20) & (cvd['ap_lo'] <= 190)]

#General Checks
cvd.isnull().sum()

# tempdata = cvdscrb[['age','ap_hi','ap_lo']]
# tempdata.describe()

df_corr = cvd.corr()
df_corr

plt.figure(figsize=(10, 10))
sns.heatmap(df_corr, annot=True, cmap='Blues') 
plt.tight_layout()
# ---------------------------------------------------------MODELING -----------------------------------------------------------

#Classification Models - Planned to be used

#Loading the independent variables in X and dependent variable in y
X = cvd.iloc[:, :11].values
y = cvd.iloc[:, 11].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)


#-------------------------------------------------------Model 1 - Logistic Regression (Linear model)---------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, y_train)

#Prediction of the Test set results
y_pred1 = classifier1.predict(X_test)
print(np.concatenate((y_pred1.reshape(len(y_pred1),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred1)
print(cm)
accuracy1 = accuracy_score(y_test, y_pred1)
print(accuracy1)

# Convert X_train from numpy array to pandas DataFrame (if necessary)
# Assuming you have a list of feature names as `feature_names`
feature_names = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholestrol', 'glu', 'smoke','alco', 'active']  # Replace with your actual feature names

# If `X_train` is a numpy array, convert it to a pandas DataFrame
if isinstance(X_train, np.ndarray):
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
else:
    X_train_df = X_train  # If X_train is already a pandas DataFrame

# To see the coefficients of the features along with the feature names:
# Access the coefficients from the model
coefficients = classifier1.coef_[0]  # For a binary classifier

# Create a DataFrame to hold the coefficients and the feature names
coeff_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coeff_df = coeff_df.reindex(coeff_df['Coefficient'].abs().sort_values(ascending=False).index)
# Print the coefficients along with the feature names
print("Coefficients of the features along with their names:")
print(coeff_df)

from sklearn.metrics import roc_curve, roc_auc_score

# Calculate the predicted probabilities of the positive class
y_prob = classifier1.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (class 1)

# Calculate FPR, TPR, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate AUC (Area Under the Curve)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line (random guess)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Logistic Regression')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

#-------------------------------------------------------Model 2 - K-Nearest Neighbors(K-NN)--------------------------------------------------------------------

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier2.fit(X_train, y_train)

# Predicting the Test set results
y_pred2 = classifier2.predict(X_test)
print(np.concatenate((y_pred2.reshape(len(y_pred2),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred2)
print(cm)
accuracy2 = accuracy_score(y_test, y_pred2)
print(accuracy2)


# Predict the probabilities of the test set using the trained KNN classifier
y_prob2 = classifier2.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (class 1)

# Calculate FPR, TPR, and thresholds using roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob2)

# Calculate AUC (Area Under the Curve)
roc_auc = roc_auc_score(y_test, y_prob2)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line (random guess)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for K-Nearest Neighbors (KNN)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


#-------------------------------------------------------Model 3 - Support Vector Machine (Kernel)--------------------------------------------------------------------

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier3 = SVC(kernel = 'rbf', random_state = 0)
classifier3.fit(X_train, y_train)

y_pred3 = classifier3.predict(X_test)
print(np.concatenate((y_pred3.reshape(len(y_pred3),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred3)
print(cm)
accuracy3 = accuracy_score(y_test, y_pred3)
print(accuracy3)

# Predict the probabilities of the test set using the trained Kernel SVM classifier
y_prob3 = classifier3.decision_function(X_test)

# Calculate FPR, TPR, and thresholds using roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob3)

# Calculate AUC (Area Under the Curve)
roc_auc = roc_auc_score(y_test, y_prob3)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line (random guess)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Kernel SVM')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
#-------------------------------------------------------Model 4 - Support Vector Machine (Linear Version)--------------------------------------------------------------------

# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier4 = SVC(kernel = 'linear', random_state = 0)
classifier4.fit(X_train, y_train)

# Predicting the Test set results
y_pred4 = classifier4.predict(X_test)
print(np.concatenate((y_pred4.reshape(len(y_pred4),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred4)
print(cm)
accuracy4 = accuracy_score(y_test, y_pred4)
print(accuracy4)

# Get decision function scores for the test set
y_scores4 = classifier4.decision_function(X_test)

# Calculate FPR, TPR, and thresholds using roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores4)

# Calculate AUC (Area Under the Curve)
roc_auc = roc_auc_score(y_test, y_scores4)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line (random guess)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Linear SVM')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
#-------------------------------------------------------Model 5 - Naive Bayes Algorithm--------------------------------------------------------------------

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier5 = GaussianNB()
classifier5.fit(X_train, y_train)

# Predicting the Test set results
y_pred5 = classifier5.predict(X_test)
print(np.concatenate((y_pred5.reshape(len(y_pred5),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred5)
print(cm)
accuracy5 = accuracy_score(y_test, y_pred5)
print(accuracy5)

# Predict the probabilities for the test set
y_prob5 = classifier5.predict_proba(X_test)[:, 1]

# Calculate FPR, TPR, and thresholds using roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob5)

# Calculate AUC (Area Under the Curve)
roc_auc = roc_auc_score(y_test, y_prob5)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line (random guess)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Naive Bayes Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

#-------------------------------------------------------Model 6 - Decision Trees Algorithm--------------------------------------------------------------------

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier6 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier6.fit(X_train, y_train)

# Predicting the Test set results
y_pred6 = classifier6.predict(X_test)
print(np.concatenate((y_pred6.reshape(len(y_pred6),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred6)
print(cm)
accuracy6 = accuracy_score(y_test, y_pred6)
print(accuracy6)

# Convert X_train from numpy array to pandas DataFrame (if necessary)
# Assuming you have a list of feature names as `feature_names`
feature_names = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholestrol', 'glu', 'smoke','alco', 'active']  # Replace with your actual feature names

# Retrieve the feature importances from the model
feature_importances = classifier6.feature_importances_

# Create a DataFrame to hold the feature names and their corresponding importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort the DataFrame in descending order based on the feature importances
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the sorted DataFrame
print("Feature importances sorted in descending order:")
print(importance_df)

# Predict the probabilities for the test set
y_prob6 = classifier6.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Calculate FPR, TPR, and thresholds using roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob6)

# Calculate AUC (Area Under the Curve)
roc_auc = roc_auc_score(y_test, y_prob6)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line (random guess)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Decision Tree Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


#-------------------------------------------------------Model 7 - Random Forest Algorithm--------------------------------------------------------------------

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier7 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier7.fit(X_train, y_train)

# Predicting the Test set results
y_pred7 = classifier7.predict(X_test)
print(np.concatenate((y_pred7.reshape(len(y_pred7),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred7)
print(cm)
accuracy7 = accuracy_score(y_test, y_pred7)
print(accuracy7)

# Convert X_train from numpy array to pandas DataFrame (if necessary)
# Assuming you have a list of feature names as `feature_names`
feature_names = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholestrol', 'glu', 'smoke','alco', 'active']  # Replace with your actual feature names

# Retrieve the feature importances from the model
feature_importances = classifier7.feature_importances_

# Create a DataFrame to hold the feature names and their corresponding importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort the DataFrame in descending order based on the feature importances
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the sorted DataFrame
print("Feature importances sorted in descending order:")
print(importance_df)


# Predict the probabilities for the test set
y_prob7 = classifier7.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Calculate FPR, TPR, and thresholds using roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob7)

# Calculate AUC (Area Under the Curve)
roc_auc = roc_auc_score(y_test, y_prob7)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line (random guess)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Random Forest Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



#-------------------------------------------------------Model 8 - Gradient Boosting Algorithm--------------------------------------------------------------------

from sklearn.ensemble import GradientBoostingClassifier
classifier8 = GradientBoostingClassifier(learning_rate=0.1)

classifier8.fit(X_train,y_train)

# Predicting the Test set results
y_pred8 = classifier8.predict(X_test)
print(np.concatenate((y_pred8.reshape(len(y_pred8),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred8)
print(cm)
accuracy8 = accuracy_score(y_test, y_pred8)
print(accuracy8)

# Convert X_train from numpy array to pandas DataFrame (if necessary)
# Assuming you have a list of feature names as `feature_names`
feature_names = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholestrol', 'glu', 'smoke','alco', 'active']  # Replace with your actual feature names

# Retrieve the feature importances from the model
feature_importances = classifier8.feature_importances_

# Create a DataFrame to hold the feature names and their corresponding importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort the DataFrame in descending order based on the feature importances
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the sorted DataFrame
print("Feature importances sorted in descending order:")
print(importance_df)

# Predict the probabilities for the test set
y_prob8 = classifier8.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Calculate FPR, TPR, and thresholds using roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob8)

# Calculate AUC (Area Under the Curve)
roc_auc = roc_auc_score(y_test, y_prob8)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line (random guess)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Gradient Boosting Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Plotting the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Gradient Boosting Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#-------------------------------------------------------Model 9 - Bagging Aggregating Clasifier Algorithm--------------------------------------------------------------------

from sklearn.ensemble import BaggingClassifier
classifier9 = BaggingClassifier(classifier4, n_estimators=12, random_state=40)
classifier9.fit(X_train, y_train)

# Predicting the Test set results
y_pred9 = classifier9.predict(X_test)
print(np.concatenate((y_pred9.reshape(len(y_pred9),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred9)
print(cm)
accuracy9 = accuracy_score(y_test, y_pred9)
print(accuracy9)

# Predict probabilities for the test set
y_prob9 = classifier9.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Calculate FPR, TPR, and thresholds using roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob9)

# Calculate AUC (Area Under the Curve)
roc_auc = roc_auc_score(y_test, y_prob9)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line (random guess)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Bagging Classifier Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
#-------------------------------------------------------Model 10 -Voting Classifier - Hard vote----------------------------------------

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Create a VotingClassifier with hard voting
classifier10 = VotingClassifier(
    estimators=[
        ('svc', classifier4),
        ('dt', classifier6),
        ('rf', classifier7),
        ('log_reg', classifier1)
    ],
    voting='hard'  # Use hard voting (majority voting)
)

# Train the VotingClassifier
classifier10.fit(X_train, y_train)

# Predicting the Test set results
y_pred_voting = classifier10.predict(X_test)

# Displaying the predicted values along with actual test set values
print(np.concatenate((y_pred_voting.reshape(len(y_pred_voting),1), y_test.reshape(len(y_test),1)),1))

# Calculate and display the confusion matrix and accuracy score
cm = confusion_matrix(y_test, y_pred_voting)
print(cm)
accuracy10 = accuracy_score(y_test, y_pred_voting)
print(accuracy10)


#-----------------------------------------------------MODEL 11 Voting Classifier - Soft vote-----------------------------------------------
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

tempclassifier = SVC(probability=True, random_state=0)
# Create a VotingClassifier with hard voting
classifier11 = VotingClassifier(
    estimators=[
        ('svc', tempclassifier),
        ('dt', classifier6),
        ('rf', classifier7),
        ('log_reg', classifier1)
    ],
    voting='soft'  # Use soft voting (minority voting)
)

# Train the VotingClassifier
classifier11.fit(X_train, y_train)

# Predicting the Test set results
y_pred_voting2 = classifier11.predict(X_test)

# Displaying the predicted values along with actual test set values
print(np.concatenate((y_pred_voting2.reshape(len(y_pred_voting2),1), y_test.reshape(len(y_test),1)),1))

# Calculate and display the confusion matrix and accuracy score
cm = confusion_matrix(y_test, y_pred_voting2)
print(cm)
accuracy11 = accuracy_score(y_test, y_pred_voting2)
print(accuracy11)


#------------------------------------------------------ MODEL 12 - ADA BOOST CLASSIFIER ----------------------------------------------------------

from sklearn.ensemble import AdaBoostClassifier
base_estimator = DecisionTreeClassifier(max_depth=1)
classifier12 = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)
classifier12.fit(X_train, y_train)
y_pred12 = classifier12.predict(X_test)
print(np.concatenate((y_pred12.reshape(len(y_pred12), 1), y_test.reshape(len(y_test), 1)), 1))
cm = confusion_matrix(y_test, y_pred12)
print(cm)
# Calculate and print the accuracy score
accuracy12 = accuracy_score(y_test, y_pred12)
print(accuracy12)

# Convert X_train from numpy array to pandas DataFrame (if necessary)
# Assuming you have a list of feature names as `feature_names`
feature_names = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholestrol', 'glu', 'smoke','alco', 'active']  # Replace with your actual feature names

# Retrieve the feature importances from the model
feature_importances = classifier12.feature_importances_

# Create a DataFrame to hold the feature names and their corresponding importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort the DataFrame in descending order based on the feature importances
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the sorted DataFrame
print("Feature importances sorted in descending order:")
print(importance_df)

# Predict probabilities for the test set
y_prob12 = classifier12.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

# Calculate FPR, TPR, and thresholds using roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob12)

# Calculate AUC (Area Under the Curve)
roc_auc = roc_auc_score(y_test, y_prob12)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line (random guess)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for AdaBoost Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
#------------------------------------------------------MODEL 13 - XGB Classifier-------------------------------------------------------------
from xgboost import XGBClassifier

# Initialize the XGBClassifier
classifier13 = XGBClassifier(random_state=42)

# Fit the model on the training set
classifier13.fit(X_train, y_train)

# Predict the test set results
y_pred13 = classifier13.predict(X_test)

# Use np.concatenate to combine the arrays along axis 1
result = np.concatenate((y_pred13.reshape(len(y_pred13), 1), y_test.reshape(len(y_test), 1)), axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred13)
print(cm)

# Calculate and print the accuracy score
accuracy13 = accuracy_score(y_test, y_pred13)
print(accuracy13)

# Predict probabilities for the test set
y_prob13 = classifier13.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

# Calculate FPR (False Positive Rate), TPR (True Positive Rate), and thresholds using roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob13)

# Calculate AUC (Area Under the Curve)
roc_auc = roc_auc_score(y_test, y_prob13)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line (random guess)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for XGBoost Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
#------------------------------------------------------MODEL 14 - Ridge Classifier-------------------------------------------------------------
from sklearn.linear_model import RidgeClassifier

# Assuming X_train, y_train, X_test, y_test, and feature_names are already defined

# Instantiate the Ridge classifier
classifier14 = RidgeClassifier(alpha=1.0, random_state=42)

# Fit the classifier to the training data
classifier14.fit(X_train, y_train)

# Predict the test set results
y_pred14 = classifier14.predict(X_test)

# Print the predictions and actual labels side by side
result = np.concatenate((y_pred14.reshape(len(y_pred14), 1), y_test.reshape(len(y_test), 1)), axis=1)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred14)
print(cm)

# Calculate and print the accuracy score
accuracy14 = accuracy_score(y_test, y_pred14)
print(accuracy14)

# Use the decision function to get the continuous predictions
y_scores14 = classifier14.decision_function(X_test)

# Calculate FPR (False Positive Rate), TPR (True Positive Rate), and thresholds using roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores14)

# Calculate AUC (Area Under the Curve)
roc_auc = roc_auc_score(y_test, y_scores14)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line (random guess)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Ridge Classifier')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
#-----------------------------------------------------Accuracy of the models------------------------------------------------------

# List of model names corresponding to the 14 models
model_names = [
    'Logistic Regression',     # Model 1
    'K-Nearest Neighbors',     # Model 2
    'Support Vector Machine (Kernel)',  # Model 3
    'Support Vector Machine (Linear)',  # Model 4
    'Naive Bayes Algorithm',   # Model 5
    'Decision Trees Algorithm', # Model 6
    'Random Forest Algorithm', # Model 7
    'Gradient Boosting Algorithm', # Model 8
    'Bagging Classifier Algorithm', # Model 9
    'Voting Classifier - Hard vote', # Model 10
    'Voting Classifier - Soft vote', # Model 11
    'AdaBoost Classifier',     # Model 12
    'XGBoost Classifier',      # Model 13
    'Ridge Classifier'         # Model 14
]

# List of accuracy values for the models (in the same order as model_names)
accuracies = [
    accuracy1*100, accuracy2*100, accuracy3*100, accuracy4*100,
    accuracy5*100, accuracy6*100, accuracy7*100, accuracy8*100,
    accuracy9*100, accuracy10*100, accuracy11*100, accuracy12*100,
    accuracy13*100, accuracy14*100
]

# Create a pandas DataFrame using the model names and accuracies
accuracy_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracies
})

# Sort the DataFrame by accuracy in descending order (optional)
accuracy_df = accuracy_df.sort_values(by='Accuracy', ascending=False)

# Print the DataFrame
print(accuracy_df)

accplt = pd.DataFrame(accuracy_df)

# Sort the DataFrame based on accuracy in descending order
df_sorted = accplt.sort_values(by='Accuracy', ascending=False)

# Create a bar chart using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Model', data=df_sorted, palette='viridis')

# Set the title and labels
plt.title('Model Accuracies')
plt.xlabel('Accuracy (%)')
plt.ylabel('Model')

# Show the plot
plt.show()
# ------------------------------------------------------PREDICTION TOOL --------------------------------------------------------------
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import dash_bootstrap_components as dbc


df = cvd  


X = df.drop('cardio', axis=1)  # Assuming 'Biopsy' is the target variable
y = df['cardio']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = GradientBoostingClassifier(learning_rate=0.1)

classifier.fit(X_train,y_train)


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.QUARTZ])

# Defining layout with 3 fields per line
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Cardiovascular Disease Prediction Tool"), width={'size': 8, 'offset': 2})),
    
    # First row with Age, Gender, and Height inputs
    dbc.Row([
        dbc.Col([
            dbc.Label("Age"),
            dbc.Input(id='age', type='number', placeholder='Enter your age')
        ], width=4),
        dbc.Col([
            dbc.Label("Gender"),
            dbc.Input(id='gender', type='number', placeholder='Enter the gender')
        ], width=4),
        dbc.Col([
            dbc.Label("Height"),
            dbc.Input(id='height', type='number', placeholder='Enter the height')
        ], width=4)
    ], className='mb-3'),
    
    # Second row with Weight, Ap_hi, and Ap_lo inputs
    dbc.Row([
        dbc.Col([
            dbc.Label("Weight"),
            dbc.Input(id='weight', type='number', placeholder='Enter the weight')
        ], width=4),
        dbc.Col([
            dbc.Label("Ap_hi"),
            dbc.Input(id='ap_hi', type='number', placeholder='Enter the ap_hi')
        ], width=4),
        dbc.Col([
            dbc.Label("Ap_lo"),
            dbc.Input(id='ap_lo', type='number', placeholder='Enter the ap_lo')
        ], width=4)
    ], className='mb-3'),

    # Third row with Cholestrol, Glucose, and Smoke inputs
    dbc.Row([
        dbc.Col([
            dbc.Label("Cholestrol"),
            dbc.Input(id='cholestrol', type='number', placeholder='Enter the cholestrol')
        ], width=4),
        dbc.Col([
            dbc.Label("Glucose"),
            dbc.Input(id='gluc', type='number', placeholder='Enter the glucose level')
        ], width=4),
        dbc.Col([
            dbc.Label("Smoke"),
            dbc.Input(id='smoke', type='number', placeholder='Enter the smoking status')
        ], width=4)
    ], className='mb-3'),

    # Fourth row with Alcohol and Active inputs
    dbc.Row([
        dbc.Col([
            dbc.Label("Alcohol"),
            dbc.Input(id='alco', type='number', placeholder='Enter the alcohol intake status')
        ], width=6),
        dbc.Col([
            dbc.Label("Active or not"),
            dbc.Input(id='active', type='number', placeholder='Enter if they are active or not')
        ], width=6)
    ], className='mb-3'),

    # Add Predict button with dbc styling
    dbc.Button('Predict', id='predict-button', color='info', className='mt-3'),
    
    # Display result in a styled dbc Alert component
    dbc.Alert(id='output-container', color='info', className='mt-3'),
], className='mt-5')

# Define callback to handle prediction
@app.callback(
    Output('output-container', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('age', 'value'),
     State('gender', 'value'),
     State('height', 'value'),
     State('weight', 'value'),
     State('ap_hi', 'value'),
     State('ap_lo', 'value'),
     State('cholestrol', 'value'),
     State('gluc', 'value'),
     State('smoke', 'value'),
     State('alco', 'value'),
     State('active', 'value')]
)
def update_output(n_clicks, age, gender, height, weight,
                  ap_hi, ap_lo, cholestrol, gluc,
                  smoke, alco, active):

    if n_clicks is None:
        return ''

    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Height': [height],
        'Weight': [weight],
        'Ap_hi': [ap_hi],
        'Ap_lo': [ap_lo],
        'Cholestrol': [cholestrol],
        'Glucose': [gluc],
        'Smoke': [smoke],
        'Alcohol': [alco],
        'Active': [active]
    })

    # Make prediction
    prediction = classifier.predict(input_data)[0]

    # Calculate accuracy on the test set (just for display, you might not use this in a real application)
    y_pred_test = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)

    # Display results based on prediction
    if prediction == 1:
        result_text = f"You have been diagnosed with Cardiovascular Disease. Accuracy on the Test Set: {accuracy:.2f}"
    else:
        result_text = f"Great news! You do not have any Cardiovascular Disease. Accuracy on the Test Set: {accuracy:.2f}"
        
    return result_text
        
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)





# ------------------------------- END -----------------------------













