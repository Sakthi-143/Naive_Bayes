# Naive_Bayes

 
# Naive Bayes Classification Model for Salary Prediction

## Overview
This repository contains a Jupyter Notebook (`Naive_Bayes_Assignment.ipynb`) that implements a classification model using Naive Bayes for predicting salary based on various features. The dataset used for this analysis includes information about individuals' demographics and employment details.

## Data Description
The dataset consists of the following features:
- `age`: Age of the individual
- `workclass`: Type of work classification 
- `education`: Level of education
- `maritalstatus`: Marital status of the individual
- `occupation`: Occupation of the individual
- `relationship`: Relationship status 
- `race`: Race of the individual
- `sex`: Gender of the individual
- `capitalgain`: Capital gain from investments
- `capitalloss`: Capital loss from investments
- `hoursperweek`: Number of hours worked per week
- `native`: Native country
- `Salary`: Salary bracket (target variable)

## Import Libraries
The notebook begins with importing necessary libraries such as NumPy, pandas, Matplotlib, Seaborn, and scikit-learn for data analysis, visualization, and modeling.

## Import Dataset
The dataset is imported into the notebook using Google Colab's file upload feature. Both training and testing datasets are loaded separately.

## Exploratory Data Analysis
Exploratory data analysis (EDA) is performed to understand the structure and characteristics of the dataset. This includes examining the shape, summary statistics, missing values, and distribution of both categorical and numerical variables.

## Data Preprocessing
Data preprocessing steps involve handling missing values, encoding categorical variables, and preparing the data for modeling. Special characters such as '?' are identified and handled appropriately.

## Feature Engineering
Feature engineering techniques are applied to transform and enhance the existing features for better model performance.

## Model Building
A Naive Bayes classification model is built using the training dataset. The model is trained to predict the salary bracket of individuals based on the provided features.

## Model Evaluation
The performance of the trained model is evaluated using various metrics such as accuracy, precision, recall, and F1-score. Cross-validation techniques may also be employed for robust evaluation.

## Conclusion
The notebook concludes with a summary of the modeling process and insights gained from the analysis. Suggestions for further improvements or experiments may also be provided.


Declare feature vector and target variable

X = SalaryData_Train.drop(['Salary'], axis=1)

y = SalaryData_Train['Salary']


Split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


Check the shape of X_train and X_test

X_train.shape, X_test.shape
((20922, 13), (8967, 13))


Feature Engineering
Display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
categorical
['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native']


Display numerical variables
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
numerical
['age', 'educationno', 'capitalgain', 'capitalloss', 'hoursperweek']
Print percentage of missing values in the categorical variables in the training set

X_train[categorical].isnull().mean()
workclass        0.0
education        0.0
maritalstatus    0.0
occupation       0.0
relationship     0.0
race             0.0
sex              0.0
native           0.0
dtype: float64
Print categorical variables with missing data

for col in categorical:
    if X_train[col].isnull().mean() > 0:
        print(col, (X_train[col].isnull().mean()))
Impute missing categorical variables with most frequent value

for df2 in [X_train, X_test]:
    df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
    df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
    df2['native'].fillna(X_train['native'].mode()[0], inplace=True)
Check missing values in categorical variables in X_train
python
Copy code
X_train[categorical].isnull().sum()
workclass        0
education        0
maritalstatus    0
occupation       0
relationship     0
race             0
sex              0
native           0
dtype: int64
Check missing values in categorical variables in X_test
python
Copy code
X_test[categorical].isnull().sum()
workclass        0
education        0
maritalstatus    0
occupation       0
relationship     0
race             0
sex              0
native           0
dtype: int64
Check missing values in X_train

X_train.isnull().sum()
age              0
workclass        0
education        0
educationno      0
maritalstatus    0
occupation       0
relationship     0
race             0
sex              0
capitalgain      0
capitalloss      0
hoursperweek     0
native           0
dtype: int64
Check missing values in X_test

X_test.isnull().sum()
age              0
workclass        0
education        0
educationno      0
maritalstatus    0
occupation       0
relationship     0
race             0
sex              0
capitalgain      0
capitalloss      0
hoursperweek     0
native           0
dtype: int64
Encode categorical variables

!pip install category_encoders

# import category encoders
import category_encoders as ce

# encode remaining variables with one-hot encoding
encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship',
                                 'race', 'sex', 'native'])

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
Feature Scaling

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Model Training

# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB

# instantiate the model
gnb = GaussianNB()

# fit the model
gnb.fit(X_train, y_train)
Predict the results

y_pred = gnb.predict(X_test)
Check accuracy score

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy score: {0:0.4f}'.format(accuracy))
Compare the train-set and test-set accuracy

y_pred_train = gnb.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
Check for overfitting and underfitting

# check null accuracy score
null_accuracy = (7407/(7407+2362))
print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
Confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
Classification metrices

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
Classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
Classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error : {0:0.4f}'.format(classification_error))
Precision

precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))
Recall

recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))
True Positive Rate

true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
False Positive Rate

false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
Specificity

specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))
Calculate class probabilities
y_pred_prob = gnb.predict_proba(X_test)[0:10]
print(y_pred_prob)
ROC - AUC

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = 1, drop_intermediate=True)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Gaussian Naive Bayes Classifier for Predicting Salaries')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()
Interpretation

ROC_AUC = roc_auc_score(y_test, y_pred1)
print('ROC AUC : {:.4f}'.format(ROC_AUC))

Cross_validated_ROC_AUC = cross_val_score(gnb, X_train, y_train, cv=5, scoring='roc_auc').mean()
print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))

scores = cross_val_score(gnb, X_train, y_train, cv = 10, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))

print('Average cross-validation score: {:.4f}'.format(scores.mean()))
