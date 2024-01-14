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
    classification_report, ConfusionMatrixDisplay, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
import warnings
warnings.filterwarnings("ignore")



pd.set_option('display.max_columns', None)
#Reading the merged dataset

df_application_record= pd.read_csv("/Users/rashmikulkarni/VT-Fall/ML1/ML-Project/application_record.csv", low_memory=False)

df_credit_record = pd.read_csv("/Users/rashmikulkarni/VT-Fall/ML1/ML-Project/credit_record.csv", low_memory=False)



# Filter df_application_record to retain only rows with IDs consistent in df_credit_record
df_application_record = df_application_record[df_application_record['ID'].isin(df_credit_record['ID'])]

# Filter df_credit_record to retain only rows with IDs consistent in df_application_record
df_credit_record = df_credit_record[df_credit_record['ID'].isin(df_application_record['ID'])]



#Replacing the NaN values for occupation type with mode of the column

mode_occupation = df_application_record['OCCUPATION_TYPE'].mode()[0]

df_application_record['OCCUPATION_TYPE'].fillna(mode_occupation, inplace=True)

df_application_record = df_application_record.drop_duplicates('ID')
df_credit_record = df_credit_record.drop_duplicates('ID')

#----------------------Merging the datasets----------------------------------------------------
df_final = df_application_record.merge(df_credit_record, on='ID', how='inner')

duplicate_ids_application = df_application_record[df_application_record.duplicated('ID')]
duplicate_ids_credit = df_credit_record[df_credit_record.duplicated('ID')]



# --------Data cleaning----------------------------------------------------------------

df_final = df_final[df_final['CNT_CHILDREN'] >= 0]

df_final['CNT_FAM_MEMBERS'] = df_final['CNT_FAM_MEMBERS'].apply(lambda x: max(0, int(x)))
df_final['AMT_INCOME_TOTAL'] = df_final['AMT_INCOME_TOTAL'].apply(lambda x: max(0, float(x)))

#------------One hot encoding--------------------------------
df_final = pd.get_dummies(df_final, columns = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE','STATUS'], drop_first=True)

#creating  a copy for regression ols model
df_final_rg = df_final
df_final_rg = df_final.drop(columns=['ID','FLAG_MOBIL','CNT_CHILDREN','DAYS_BIRTH'])

#----------------------Creating a new continuous variable called EMPLOYABILITY-INCOME RATIO----------------------------------------------------

df_final_rg['employment_to_income'] = (df_final_rg['DAYS_EMPLOYED'] / df_final_rg['AMT_INCOME_TOTAL'])
X = df_final_rg.drop(['AMT_INCOME_TOTAL','DAYS_EMPLOYED','employment_to_income'],axis = 1)
y = df_final_rg['employment_to_income']

X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, shuffle = True, random_state=5808)

removed_features = []

display_table = pd.DataFrame(columns=["AIC", "BIC", "Adj. R2", "P-Value", "R2"])

while True:
    # print("After removing\n", max_p_value_feature)
    # print(model_regression.summary())
    model_regression = sm.OLS(y_train, X_train).fit()

    # Find the feature with the maximum p-value
    max_p_value_feature = model_regression.pvalues.idxmax()

    # Check if the maximum p-value is greater than 0.05
    if model_regression.pvalues.max() >= 0.01:
        # Record the statistics before dropping the feature
        display_table.loc[max_p_value_feature, "AIC"] = model_regression.aic.round(3)
        display_table.loc[max_p_value_feature, "BIC"] = model_regression.bic.round(3)
        display_table.loc[max_p_value_feature, "Adj. R2"] = model_regression.rsquared_adj.round(3)
        display_table.loc[max_p_value_feature, "P-Value"] = model_regression.pvalues[max_p_value_feature].round(3)
        display_table.loc[max_p_value_feature, "R2"] = model_regression.rsquared.round(3)

        # Drop the feature with the maximum p-value
        print(model_regression.summary())
        print("After removing\n", max_p_value_feature)
        X_train = X_train.drop(max_p_value_feature, axis=1)
        X_test = X_test.drop(max_p_value_feature, axis=1)
        removed_features.append(max_p_value_feature)


    else:
        # If the maximum p-value is not greater than 0.01, break the loop
        break

print("Final ols model",model_regression.summary())

selected_features_reg = X_train.columns

print("\nSelected Features after ols are:", selected_features_reg)
print("\nFeatures that are Eliminated after ols are:", removed_features)
print(display_table)

final_model_regression = sm.OLS(y_train, X_train).fit()
print("Final model regression OLS:", final_model_regression.summary())
y_predicted_ols = final_model_regression.predict(X_test)

confidence_interval = round(final_model_regression.conf_int(),3)
print("Confidence Interval:\n", confidence_interval)



# Invert the scaling transformation on y_test_hat
y_test_hat = y_predicted_ols * y.std() + y.mean()

tr_range = [p for p in range(len(y_train))]
tst_range = [p for p in range(y_test.shape[0])]
plt.plot(tr_range, y_train, label='Training Employment-Income ratio')
plt.plot(tst_range, y_test, label='Testing Employment-Income ratio')
plt.plot(tst_range,y_predicted_ols,label='Predicted Employment-Income ratio')
plt.title('Regression Results in one plot')
plt.xlabel('Observations')
plt.ylabel('Employment-Income Ratio')
plt.legend(loc='upper right')
plt.show()



ols_table = PrettyTable()

# Define the table headers
ols_table.field_names = ["Metric", "Value"]


ols_table.add_row(["Mean Squared Error (MSE) for OLS", round(mean_squared_error(y_test, y_predicted_ols), 3)])
ols_table.add_row(["R-squared", round(final_model_regression.rsquared, 3)])
ols_table.add_row(["Adjusted R-squared", round(final_model_regression.rsquared_adj, 3)])
ols_table.add_row(["AIC", round(final_model_regression.aic, 3)])
ols_table.add_row(["BIC", round(final_model_regression.bic, 3)])

print("METRICS AFTER MULTIPLE REGRESSION")
print(ols_table)









