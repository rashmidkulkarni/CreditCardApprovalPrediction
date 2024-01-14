import pandas as pd
import numpy as np
import seaborn as seaborn
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import statsmodels.api as sm
from sklearn.decomposition import PCA, TruncatedSVD
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, \
    classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier

pd.set_option('display.max_columns', None)
#Reading the merged dataset

df_application_record= pd.read_csv("/Users/rashmikulkarni/VT-Fall/ML1/ML-Project/application_record.csv", low_memory=False)
print(df_application_record.head())

df_credit_record = pd.read_csv("/Users/rashmikulkarni/VT-Fall/ML1/ML-Project/credit_record.csv", low_memory=False)
print("Information about record df",df_application_record.info())
print("Unique values in record df",df_application_record.nunique())

# Counting the number of unique IDs that are consistent between both datasets
consistent_ids = df_application_record['ID'].isin(df_credit_record['ID'])
unique_consistent_ids = df_application_record[consistent_ids]['ID'].nunique()
print(f"Number of unique IDs that are consistent between both datasets: {unique_consistent_ids}")

# Filter df_application_record to retain only rows with IDs consistent in df_credit_record
df_application_record = df_application_record[df_application_record['ID'].isin(df_credit_record['ID'])]

# Filter df_credit_record to retain only rows with IDs consistent in df_application_record
df_credit_record = df_credit_record[df_credit_record['ID'].isin(df_application_record['ID'])]

print("Confirming the number of ids for application record dataset",df_application_record['ID'].nunique())
print("Confirming the number of ids for credit record dataset",df_credit_record['ID'].nunique())

# Checking for null values
print("Nulls in application record dataset",df_application_record.isna().sum())
print("Nulls in credit record dataset",df_credit_record.isna().sum())

print("% of null values in occupation type  ",df_application_record.isna().sum()/df_application_record.shape[0])

#Replacing the NaN values for occupation type with mode of the column

mode_occupation = df_application_record['OCCUPATION_TYPE'].mode()[0]
print(f"Mode:{mode_occupation}")
df_application_record['OCCUPATION_TYPE'].fillna(mode_occupation, inplace=True)


print("% of null values in occupation type  ",df_application_record.isna().sum()/df_application_record.shape[0])

df_application_record = df_application_record.drop_duplicates('ID')
df_credit_record = df_credit_record.drop_duplicates('ID')

#----------------------Merging the datasets----------------------------------------------------
df_final = df_application_record.merge(df_credit_record, on='ID', how='inner')

print('Unique ids', df_final.nunique())
print('Total dataset count', df_final.shape[0])

duplicate_ids_application = df_application_record[df_application_record.duplicated('ID')]
duplicate_ids_credit = df_credit_record[df_credit_record.duplicated('ID')]

print('duplicates in the application set',duplicate_ids_application)
print('duplicates in the credit set',duplicate_ids_credit)


print('Unique ids', df_final.nunique())
print('Total dataset count', df_final.shape[0])


df_final.head(10)


df_final.loc[(df_final['STATUS'] == 'X') | (df_final['STATUS'] == 'C') | (df_final['STATUS'] == '0'), 'response'] = 1
df_final.loc[(df_final['STATUS'] == '1') | (df_final['STATUS'] == '2') | (df_final['STATUS'] == '3') | (df_final['STATUS'] == '4') | (df_final['STATUS'] == '5'), 'response'] = 0