import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import ipaddress
from sklearn.ensemble import IsolationForest
import pickle


# Converting IP Addresses To Integers
def ip_to_int(ip):
    return int(ipaddress.ip_address(ip))


# A function to choose classifiers
def make_pipeline(classifier_key):
    if classifier_key in classifiers:
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifiers[classifier_key])
        ])
        return clf
    else:
        raise ValueError(f"Classifier {classifier_key} is not defined")


# Load The Dataset
data = pd.read_csv('data/rba-dataset.csv', chunksize=150000)


for chunk in data:


    # Calculating The Exact Hour of Day
    chunk['Login Hour'] = pd.to_datetime(chunk['Login Timestamp']).dt.hour

    # Converting Booleans To Integers
    chunk['Is Account Takeover'] = chunk['Is Account Takeover'].astype(np.uint8)
    chunk['Is Attack IP'] = chunk['Is Attack IP'].astype(np.uint8)
    chunk['Login Successful'] = chunk['Login Successful'].astype(np.uint8)

    # Dropping Unneeded Columns
    chunk = chunk.drop(columns=["Round-Trip Time [ms]", 'Region', 'City', 'Login Timestamp', 'index'])

    # Converting Strings To Integers
    chunk['User Agent String'], _ = pd.factorize(chunk['User Agent String'])
    chunk['Browser Name and Version'], _ = pd.factorize(chunk['Browser Name and Version'])
    chunk['OS Name and Version'], _ = pd.factorize(chunk['OS Name and Version'])

    # Converting IP Addresses To Integers
    chunk['IP Address'] = chunk['IP Address'].apply(ip_to_int)

    # Encoding Categorical & Numerical Variables
    categorical_cols = ['Country', 'Device Type']
    numeric_cols = ['ASN', 'Login Hour', 'IP Address', 'User Agent String', 'Browser Name and Version',
                    'OS Name and Version']

    # Splitting the chunkset
    features = chunk.drop(['Is Attack IP', 'Is Account Takeover'], axis=1)
    labels = chunk['Is Account Takeover']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    # Classifiers
    classifiers = {
        'svm': SVC(probability=True)
    }

    classifier_key = 'svm'

    pipeline = make_pipeline(classifier_key)
    pipeline.fit(X_train, y_train)

    # Evaluation
    predictions = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, probs)


    with open('bin/model.pkl', 'wb') as file:
        pickle.dump(pipeline, file)

