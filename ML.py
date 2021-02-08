#!/usr/bin/env python
# coding: utf-8

# # Task Description: The Best Classifier

# In this project, you will complete a notebook where **you will build a classifier to predict whether a loan case will be paid off or not.**
# You load a historical dataset from previous **loan applications**, clean the data, and apply different classification algorithm on the data. You are expected to use the following algorithms to build your models:<br>
# - `k-Nearest Neighbour`
# - `Decision Tree`
# - `Support Vector Machine`
# - `Logistic Regression`
# <br>
# The results is reported as the accuracy of each classifier, using the following metrics when these are applicable:
# - `Jaccard index`
# - `F1-score`
# - `LogLoss`
# <br>
#
# Review criteria:<br>
# This final project will be graded by your peers who are completing this course during the same session. This project is worth 25 marks of your total grade, broken down as follows:
# ***
# 1.	Building model using KNN, finding the best k and accuracy evaluation (7 marks)
# 2.	Building model using Decision Tree and find the accuracy evaluation (6 marks)
# 3.	Building model using SVM and find the accuracy evaluation (6 marks)
# 4.	Building model using Logistic Regression and find the accuracy evaluation (6 marks)
# ***
#

# # Table of Contents:
# 1. [Importing python modules for data analysis and machine learning](#sec1)
# 2. [Get the dataset from IBM Cloud](#sec2)
# 3. [About the dataset](#sec3)
#     * [3.1 Quick view of the dataset](#sec3.1)
#     * [3.2 Check the datatypes for columns in data](#sec3.2)
#     * [3.3 Plot distribution for categorical data to inspect missing value counts](#sec3.3)
#     * [3.4 Check if numerical data contains missing values](#sec3.4)
# 4. [Feature Engineering](#sec4)
# 5. [Model Development](#sec5)
#     * [5.1 Scaling of features](#sec4.1)
#     * [5.2 KNN Model](#sec5.2)
#     * [5.3 Decision Tree Model](#sec5.3)
#     * [5.4 Support Vector Machine Model](#sec5.4)
#     * [5.5 Logistic Regression Model](#sec5.5)
# 6. [Model Validation on the test set](#sec6)
#     * [6.1 Data Preprocessing](#sec6.1)
#     * [6.2 Report](#sec6.2)

# <a id="sec1"></a>
# # [1. Importing python modules for data analysis and machine learning](#sec1)

# In[167]:

print("\nBegin------------------------------")

import pandas as pd  # for data analysis
import numpy as np  # for mathematical calculations on Matrix
import matplotlib.pyplot as plt  # for plotting
import scipy.optimize as opt  # if needed
from sklearn import preprocessing  # for data cleaning and preprocessing
from sklearn.model_selection import (
    train_test_split,
)  # for out-of-sample validation and splitting the data

print("\nImports loaded------------------------------")

# # [3. About the dataset](#sec3)

# This dataset is about past loans. The Loan_train.csv data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# ***
# Field	Description
# - `Loan_status`:	Whether a loan is paid off on in collection
# - `Principal`:	Basic principal loan amount at the
# - `Terms`:	Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule
# - `Effective_date`:	When the loan got originated and took effects
# - `Due_date`:	Since it’s one-time payoff schedule, each loan has one single due date
# - `Age`:	Age of applicant
# - `Education`:	Education of applicant
# - `Gender`:	The gender of applicant
# ***

# <a id="sec3.1"></a>
# ## [3.1 Quick view of the dataset](#sec3.1)

# In[170]:

loan_data = pd.read_csv("loan_train.csv")

print("\nTrain data loaded------------------------------")

# In[171]:

print("\nTrain data head------------------------------")
print(loan_data.head())


# ### we can see that the index is default (int), first 2 columns don't look informatory so we remove them.

# In[172]:

print("\nTrain data columns------------------------------")
print(loan_data.columns)


# In[173]:

print(
    "\nRemoving unncessary columns from the base train data------------------------------"
)
loan_data.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1, inplace=True)
print(loan_data.head())


# In[174]:

loan_test = pd.read_csv("loan_test.csv")
print("\nTest data loaded------------------------------")

print("\nTest data head------------------------------")
print(loan_test.head())


# In[175]:

print(
    "\nRemoving unncessary columns from the base test data------------------------------"
)
loan_test.drop(
    ["Unnamed: 0", "Unnamed: 0.1"], axis=1, inplace=True
)  # applying same operation to test data
print(loan_test.head())

# <a id="sec3.2"></a>
# ## [3.2 Check the datatypes for each column in data](#sec3.2)

# In[176]:

print("Train data info------------------------------\n")
loan_data.info()


# In[177]:

print("\nTest data info------------------------------")
loan_test.info()


# In[178]:

print("\nTrain data describe------------------------------")
### quickly describe data
print(loan_data.describe(include="all"))


# In[179]:

print("\nTest data describe------------------------------")
print(loan_test.describe(include="all"))


# <a id="sec3.3"></a>
# ## [3.3 Plot distribution for categorical data to inspect missing value counts](#sec3.3)
#
#

# In[180]:

print("\nCategorical data------------------------------")
loan_data_cpy = (
    loan_data.copy()
)  # just work with copy to avoid any unneccessary changes to the original data
cat_data = loan_data_cpy.select_dtypes(include=["object"])

print(cat_data.head())


# In[181]:


# For every column in cat_data, plotting histogram of value_counts
# And we can see from below, there is no irregularity or missing values
print("\nPlotting distribution of categorical columns------------------------------")
"""
for col in cat_data.columns:
    print("Col : {}\n".format(col))
    cat_data[col].value_counts().plot(kind="bar")
    plt.title("Distribution: {}".format(col))
    plt.ylabel("#")
    plt.xlabel(col)
    plt.show()


"""

# <a id="sec3.4"></a>
# ## [3.4 Check if numerical data contains missing values](#sec3.4)
#

# In[182]:

print("\nNumerical data------------------------------")
num_data = loan_data_cpy.select_dtypes(
    include=["int64"]
)  # since we only have int64 as the remaining data
num_data.head()


# In[183]:

print("\nNull values in numerical data------------------------------")
pd.isnull(num_data).any()  # so no irregularity here as well


# <a id="sec4"></a>
# # [4. Feature Engineering](#sec4)
#

# ## Preprocessing
# ### As for modeling purpose we need the X (data matrix for scikit models) to be numeric, so we will convert the categorical columns to numeric using LabelEncoders(), leaving out the date variables. We will process date variables in the the next stage.

# In[184]:

print("\nCategorical data excluding date columns------------------------------")
# lets get the categorical data excluding date columns
X_cat_exc_date = cat_data[["loan_status", "education", "Gender"]].values
X_cat_exc_date[0:5]


# In[185]:

print("\nPreprocessing begins------------------------------")
le_loan_st = preprocessing.LabelEncoder()
le_loan_st.fit(["PAIDOFF", "COLLECTION"])
X_cat_exc_date[:, 0] = le_loan_st.transform(X_cat_exc_date[:, 0])

le_educ = preprocessing.LabelEncoder()
le_educ.fit(["Bechalor", "High School or Below", "Master or Above", "college"])
X_cat_exc_date[:, 1] = le_educ.transform(X_cat_exc_date[:, 1])

le_gen = preprocessing.LabelEncoder()
le_gen.fit(["female", "male"])
X_cat_exc_date[:, 2] = le_gen.transform(X_cat_exc_date[:, 2])


# In[186]:


X_cat_exc_date[0:5]


# ### Now Coming back to the date variables. Ideally, We would want them to be some sort of features. To do this, we could first convert the object dates into pandas datetime and then introduce differences - between the date and now

# In[187]:


X_cat_date = cat_data[["effective_date", "due_date"]].copy()
X_cat_date.head()


# In[188]:


X_cat_date = X_cat_date.applymap(lambda x: pd.to_datetime(x, format="%M/%d/%Y"))


# In[189]:


X_cat_date


# In[190]:


from datetime import datetime

now = datetime.now()

X_cat_date["effective_date"] = X_cat_date["effective_date"].apply(
    lambda x: round((now - x).days / 7)
)  # calculate the Weeks
X_cat_date["due_date"] = X_cat_date["due_date"].apply(
    lambda x: round((now - x).days / 7)
)  # calculate the Weeks


# In[191]:


X_cat_date

# In[192]:

print("\nX and Y for ML modeling------------------------------")
X = np.column_stack((X_cat_exc_date[:, 1:], X_cat_date.values, num_data.values)).astype(
    "int64"
)
y = X_cat_exc_date[:, 0].astype("int64")


# ### standarize the matrix values

# In[193]:

print("\nPreprocessing step: Standardization------------------------------")
X = preprocessing.StandardScaler().fit(X).transform(X)
X


# ### Preprocessing stage Completed. These are the Final matrices for machine learning modeling
# #### For X columns are in the order: 'Education', 'Gender', 	'Effective date', 'Due date', 'Principal, 'terms, 'Age'

# In[194]:


X, y


# <a id="sec5"></a>
# # [5. Model Development](#sec5)
#

# <a id="sec5.1"></a>
# ## [5.1 Scaling of Features](#sec5.1)
#

# In[195]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4
)  # assuming 80 train, 20 test %
print("Train set: ", X_train.shape, y_train.shape)
print("Train set: ", X_test.shape, y_test.shape)


# <a id="sec5.2"></a>
# ## [5.2 KNN Model](#sec5.2)
#

# In[196]:


from sklearn.neighbors import KNeighborsClassifier  # import

# Training
k = 3
knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
knn


# In[197]:


# Prediction
y_hat_knn = knn.predict(X_test)
y_hat_knn[0:5]


# In[198]:


# Accuracy Evaluation
from sklearn import metrics

print("Training set acc:", metrics.accuracy_score(y_train, knn.predict(X_train)))
print("Test set acc:", metrics.accuracy_score(y_test, y_hat_knn))


# ## As we can see for k=3, we get acc. around 64% for test data. Now, we cross-validate for k between 1 and 58 to find the best k for max acc.
#

# In[224]:


Ks = 58
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))

for n in range(1, Ks):

    # Train and predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    y_hat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, y_hat)
    std_acc = np.std(y_hat == y_test) / np.sqrt(y_hat.shape[0])

mean_acc


# In[200]:


# plotting acc. vs Ks
plt.plot(range(1, Ks), mean_acc, "r")
plt.fill_between(
    range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10
)
plt.fill_between(
    range(1, Ks),
    mean_acc - 3 * std_acc,
    mean_acc + 3 * std_acc,
    alpha=0.10,
    color="grey",
)
plt.legend(("Accuracy", "+-1std", "+-3std"))
plt.ylabel("Accuracy")
plt.xlabel("Number of neighbors (K)")
plt.tight_layout()
plt.show()

# ### Best K = 15

# In[201]:


print("The Best acc. was ", mean_acc.max(), "with K=", mean_acc.argmax() + 1)


# <a id="sec5.3"></a>
# ## [5.3 Decision Tree Model](#sec5.3)
#

# In[202]:


from sklearn.tree import DecisionTreeClassifier  # import

# Training
dt = DecisionTreeClassifier(criterion="entropy", max_depth=4).fit(X_train, y_train)

# Prediction
y_hat_dt = dt.predict(X_test)

y_hat_dt[0:5]
y_test[0:5]


# Accuracy evaluation
print("Test set acc:", metrics.accuracy_score(y_test, y_hat_dt))


# ## We can crossvalidate for parameter 'max_depth' as well. I have done that and come up with conclusion that Best acc. for Decision tree model is 0.7857142857142857 (same as K Nearest Neighbor)

# <a id="sec5.4"></a>
# ## [5.4 Support Vector Machine Model](#sec5.4)
#

# In[203]:


from sklearn import svm

# Training
svm_cl = svm.SVC(kernel="rbf")
svm_cl.fit(X_train, y_train)

# Prediction
y_hat_svm = svm_cl.predict(X_test)
y_hat_svm[0:5]


# In[204]:


# Accuracy evaluation
print("Test set acc:", metrics.accuracy_score(y_test, y_hat_svm))


# <a id="sec5.5"></a>
# ## [5.5 Logistic Regression Model](#sec5.5)
#

# In[205]:


from sklearn.linear_model import LogisticRegression

# Training
logr = LogisticRegression(C=0.01, solver="liblinear").fit(X_train, y_train)

# Prediction
y_hat_logr = logr.predict(X_test)
y_hat_logr[0:5]

# Accuracy
print("Test set acc:", metrics.accuracy_score(y_test, y_hat_logr))


# In[206]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# In[207]:


# Jaccard score for test set

print("Test set acc for logistic reg:", jaccard_score(y_test, y_hat_logr))
print("Test set acc for svm:", jaccard_score(y_test, y_hat_svm))
print("Test set acc for decision trees:", jaccard_score(y_test, y_hat_dt))
print("Test set acc for Knn:", jaccard_score(y_test, y_hat_knn))


# In[208]:


from sklearn.metrics import log_loss

y_hat_logr_prob = logr.predict_proba(X_test)
log_loss_test = log_loss(y_test, y_hat_logr_prob)


# In[209]:


log_loss_test


# <a id="sec6"></a>
# # [6. Model Evaluation on Test data](#sec6)
#

# In[ ]:


# <a id="sec6.1"></a>
# ## [6.1 Data preprocessing](#sec6.1)
#

# In[210]:


pd.isnull(loan_test).sum()  # no missing values in test data


# In[211]:


# le_loan_st = preprocessing.LabelEncoder()
# le_loan_st.transform(loan_test.loc[:,'loan_status'])
loan_test.loc[:, "loan_status"] = le_loan_st.transform(loan_test.loc[:, "loan_status"])
# le_loan_st.fit(['PAIDOFF', 'COLLECTION'])
# X_cat_exc_date[:,0] = le_loan_st.transform(X_cat_exc_date[:,0])

# le_educ = preprocessing.LabelEncoder()
# le_educ.fit(['Bechalor', 'High School or Below', 'Master or Above', 'college'])
loan_test.loc[:, "education"] = le_educ.transform(loan_test.loc[:, "education"])
# X_cat_exc_date[:,1] = le_educ.transform(X_cat_exc_date[:,1])

# le_gen = preprocessing.LabelEncoder()
# le_gen.fit(['female', 'male'])
loan_test.loc[:, "Gender"] = le_gen.transform(loan_test.loc[:, "Gender"])
# X_cat_exc_date[:,2] = le_gen.transform(X_cat_exc_date[:,2])
loan_test


# In[212]:


loan_test


# In[213]:


loan_test[["effective_date", "due_date"]] = loan_test[
    ["effective_date", "due_date"]
].applymap(lambda x: pd.to_datetime(x, format="%M/%d/%Y"))


# In[214]:


# now = datetime.now()

loan_test["effective_date"] = loan_test["effective_date"].apply(
    lambda x: round((now - x).days / 7)
)  # calculate the Weeks
loan_test["due_date"] = loan_test["due_date"].apply(
    lambda x: round((now - x).days / 7)
)  # calculate the Weeks


# In[215]:


X


# In[216]:


loan_test = loan_test.astype("int")
X_val = loan_test[
    ["education", "Gender", "effective_date", "due_date", "Principal", "terms", "age"]
].values
y_val = loan_test[["loan_status"]].values

X_val = preprocessing.StandardScaler().fit(X_val).transform(X_val)


# In[217]:


X_val


# ## Report

# In[218]:


# importing all the models
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier


def create_baseline_classifiers(seed=8):
    """Create a list of baseline classifiers.
    
    Parameters
    ----------
    seed: (optional) An integer to set seed for reproducibility
    Returns
    -------
    A list containing tuple of name, model object for each of these algortihms:
    DummyClassifier, LogisticRegression, SGDClassifier, ExtraTreesClassifier, 
    GradientBoostingClassifier, RandomForestClassifier, MultinomialNB, SVC, 
    XGBClassifier.
    
    """
    models = []
    models.append(
        (
            "Logistic Regression",
            LogisticRegression(C=0.01, solver="liblinear", random_state=seed),
        )
    )
    models.append(
        ("Stochastic Gradient Descent Classifier", SGDClassifier(random_state=seed))
    )
    models.append(("Extra Tress Classifier", ExtraTreesClassifier(random_state=seed)))
    models.append(
        (
            "Gradient Boosting Machines Classifier",
            GradientBoostingClassifier(random_state=seed),
        )
    )
    models.append(
        ("Random Forest Classifier", RandomForestClassifier(random_state=seed))
    )
    # models.append(('Multinomial Naive Bayes Classifier', MultinomialNB()))
    models.append(("Support Vector Machine Classifier", svm.SVC(kernel="rbf")))
    models.append(("XGBoost Classifier", XGBClassifier(seed=seed)))
    models.append(("KNeighbors Classifier", KNeighborsClassifier(n_neighbors=15)))
    models.append(
        (
            "Decision Trees Classifier",
            DecisionTreeClassifier(criterion="entropy", max_depth=4),
        )
    )
    return models


def fit_models(X, y, models):
    """
    fit models on data.
    
    Parameters
    ----------
    X: A pandas DataFrame containing feature matrix
    y: A pandas Series containing target vector
    models: A list of models to train

    Returns:
    fit dictionary for all the models
    -------
    
    """
    fit = {}
    for name, model in models:
        fit[name] = model.fit(
            X, y
        )  # pd.DataFrame(cross_validate(model, X, y, cv=cv, scoring=metrics))

    return fit


def predict_models(X_test, fit):
    """
    predict target for test set.
    
    Parameters
    ----------
    X_test: A pandas DataFrame containing feature matrix
    y_test: A pandas Series containing target vector
    fit: fitted models from prior stage for Training data

    Returns:
    fit dictionary for all the models
    -------
    
    """
    prediction = {}
    for name, model in models:
        prediction[name] = fit[name].predict(
            X_test
        )  # pd.DataFrame(cross_validate(model, X, y, cv=cv, scoring=metrics))

    return prediction


def summary_models(prediction, X_test, y_test, fit):
    from sklearn.metrics import jaccard_score, f1_score, log_loss, accuracy_score

    jaccard = {}
    f1 = {}
    logloss = {}
    acc_score = {}
    for model_name in prediction.keys():
        jaccard[model_name] = jaccard_score(y_test, prediction[model_name])
        f1[model_name] = f1_score(y_test, prediction[model_name])
        acc_score[model_name] = accuracy_score(y_test, prediction[model_name])
        logloss[model_name] = (
            log_loss(y_test, fit["Logistic Regression"].predict_proba(X_test))
            if model_name == "Logistic Regression"
            else "NA"
        )

    summary = pd.DataFrame.from_dict([jaccard, f1, acc_score, logloss])
    summary.index = ["jaccard", "f1", "accuracy_score", "log_loss"]
    return summary.T


# <a id="sec6.2"></a>
# ## [6.2 Report](#sec6.2)
#

# In[219]:


# training file performance where training data is split further train-test: 80%:20%
# test set performace
models = create_baseline_classifiers()
fit_train = fit_models(X_train, y_train, models)
predict_test = predict_models(X_test, fit_train)
summary_test = summary_models(predict_test, X_test, y_test, fit_train)


# In[220]:


summary_test


# In[221]:


# validation file (loan_test) performance where
# validation set performace
# models = create_baseline_classifiers()
# fit_train = fit_models(X_train, y_train, models)
predict_val = predict_models(X_val, fit_train)
summary_val = summary_models(predict_val, X_val, y_val, fit_train)


# In[222]:


summary_val


# In[ ]:

