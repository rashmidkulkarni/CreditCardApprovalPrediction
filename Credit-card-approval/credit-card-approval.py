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
print(df_application_record.head())

df_credit_record = pd.read_csv("/Users/rashmikulkarni/VT-Fall/ML1/ML-Project/credit_record.csv", low_memory=False)
print("Information about record df",df_application_record.info())

print("Unique values in record df",df_application_record.nunique())

df_application_record = df_application_record.drop_duplicates(subset = 'ID', keep = False)

print("Information about record df",df_application_record.info())

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

print("Information about record df to check occupation type change",df_application_record.info())


print("% of null values in occupation type  ",df_application_record.isna().sum()/df_application_record.shape[0])

# Display the rows that have duplicates in any column
all_duplicates_record = df_application_record[df_application_record.duplicated()]
print("Rows with Duplicates in All Columns record dataset:",all_duplicates_record.shape[0])
all_duplicates_credit = df_credit_record[df_credit_record.duplicated()]
print("Rows with Duplicates in All Columns credit dataset:",all_duplicates_credit.shape[0])

# Display the rows that have duplicates in any column

print("Credit Record unique",df_credit_record.nunique())
df_application_record = df_application_record.drop_duplicates('ID')
df_credit_record = df_credit_record.drop_duplicates('ID')

#----------------------Merging the datasets----------------------------------------------------
df_final = df_application_record.merge(df_credit_record, on='ID', how='inner')

print("Information about final df",df_final.info())
print('Unique ids', df_final.nunique())
print('Total dataset count', df_final.shape[0])

duplicate_ids_application = df_application_record[df_application_record.duplicated('ID')]
duplicate_ids_credit = df_credit_record[df_credit_record.duplicated('ID')]

print('duplicates in the application set',duplicate_ids_application)
print('duplicates in the credit set',duplicate_ids_credit)


print('Unique ids', df_final.nunique())
print('Total dataset count', df_final.shape[0])


df_final.head(10)

#EDA for the data
#Plotting the graph to know the income for each gender

seaborn.boxplot(data=df_final, x="CODE_GENDER", y="AMT_INCOME_TOTAL",hue="CODE_GENDER")
plt.title("Income of a person based on gender")
plt.xlabel("Gender")
plt.ylabel("Income")
plt.legend(title="Gender vs Income",loc='upper right')
plt.show()

#Plotting thr graph of education type to gender
plt.figure(figsize=(12,12))
seaborn.countplot(data=df_final, x="NAME_EDUCATION_TYPE", hue ="CODE_GENDER")
plt.xticks(rotation=20)
plt.title("Education type based on gender")
plt.legend(title="Gender vs Income")
plt.show()

#Plotting the count of people having different types of houses
plt.figure(figsize=(12, 8))
sns.countplot(data=df_final, x="NAME_HOUSING_TYPE")
plt.title("Count of people having different types of houses")
plt.xlabel("Housing Type")
plt.ylabel("Count")
plt.show()

plt.show()
# Observation: Many people own houses.

#Plotting the count of people having different types of income types
plt.figure(figsize=(12,8))
seaborn.countplot(data=df_final, x="NAME_INCOME_TYPE",hue ="CODE_GENDER")
plt.xlabel("Income Type")
plt.title("Count of people having different types of income types")
plt.show()

#Observation: Max people work and no on is a student in the dataset

#Plotting the count of people having a car
plt.figure(figsize=(12,8))
seaborn.countplot(data=df_final, x="FLAG_OWN_CAR",hue="CODE_GENDER")
plt.title("Count of people owning a car")
plt.xlabel("Number of people owning a car")
plt.legend(loc="upper left")
plt.show()



#------------------Adding response variable considering STATUS variable------------------------------------------------
df_final.loc[(df_final['STATUS'] == 'X') | (df_final['STATUS'] == 'C') | (df_final['STATUS'] == '0'), 'response'] = 1
df_final.loc[(df_final['STATUS'] == '1') | (df_final['STATUS'] == '2') | (df_final['STATUS'] == '3') | (df_final['STATUS'] == '4') | (df_final['STATUS'] == '5'), 'response'] = 0
print(df_final.head(5))



#--------Data cleaning----------------------------------------------------------------
plt.figure(figsize=(20, 15))
correlation_matrix = df_final.corr().round(1)
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu',xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns)
plt.title('Correlation Matrix to show correlation between features.')
plt.show()


df_final = df_final[df_final['CNT_CHILDREN'] >= 0]
df_final['CNT_FAM_MEMBERS'] = df_final['CNT_FAM_MEMBERS'].apply(lambda x: max(0, int(x)))
df_final['AMT_INCOME_TOTAL'] = df_final['AMT_INCOME_TOTAL'].apply(lambda x: max(0, float(x)))

#------------One hot encoding--------------------------------
df_final = pd.get_dummies(df_final, columns = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'], drop_first=True)
print('df_final columns after one hot encoding',df_final.columns)
print("Dataset after one hot encoded::::::::",df_final.head(5))

#--------Removing columns with no information like ID--------------------------------
df_final = df_final.drop(columns=['ID','STATUS','FLAG_MOBIL','CNT_CHILDREN','DAYS_BIRTH'])#RESPONSE BASED ON STATUS


plt.figure(figsize=(10,10))
sns.countplot(x='response', data=df_final, palette='viridis',hue = 'response')
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.legend()
plt.show()


# Applying SMOTE
df_final = df_final.reset_index()
X = df_final.drop(columns=['response'], axis=1)
y = df_final['response']
print("y value count",y.value_counts())


smote = SMOTE(random_state=5805)
X_smote_resampled, y_smote_resampled = smote.fit_resample(X, y)

df_final = pd.concat([pd.DataFrame(X_smote_resampled, columns=X.columns), pd.DataFrame({'response': y_smote_resampled})], axis=1)

class_distribution_after = df_final['response'].value_counts()
print("\nClass distribution after SMOTE:")
print(class_distribution_after)

# Plotting the count plot before and after SMOTE
plt.figure(figsize=(10,10))
sns.countplot(x='response', data=df_final, palette='viridis',hue = 'response')
plt.title('Class Distribution After SMOTE')
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.legend()
plt.show()

#-------------------Boxplot for Outliers--------------------------

plt.figure(figsize=(25, 10))
columns_to_include = [col for col in df_final.columns if col != 'response']
sns.boxplot(data=df_final[columns_to_include], orient='h')
plt.xlabel("Values")
plt.title("Boxplot for Outlier Detection")
plt.grid(True)
plt.show()



#---------- Standardisation--------------------------------------------------------------------------------------------
standardize_features = ['AMT_INCOME_TOTAL','DAYS_EMPLOYED','CNT_FAM_MEMBERS','MONTHS_BALANCE']


def standardize(df):
    standardized_df = round((df - df.mean()) / df.std(), 2)
    return standardized_df

print('Dataset after scaling::::::::')
df_final[standardize_features] = standardize(df_final[standardize_features])
print(df_final.head(5))

#---------Aplying IQR--------------------------------


# def filter_outliers(df, column_name):
#     q1, q3 = df[column_name].quantile([0.25, 0.75])
#     iqr = q3 - q1
#     lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
#     filtered_df = df[(df[column_name] > lower_bound) & (df[column_name] < upper_bound)]
#
#     if not filtered_df.empty:
#         return filtered_df
#     else:
#         print(f"Warning: No values remaining after filtering for {column_name}")
#         return df
#
# to_consider = ['AMT_INCOME_TOTAL','DAYS_EMPLOYED','FLAG_WORK_PHONE','FLAG_EMAIL','MONTHS_BALANCE']
# for feature in to_consider:
#     df_final = filter_outliers(df_final, feature)


#Trying feature importance using random forest
df_final = df_final[: : 2]
df_final.drop(['index'], axis=1, inplace=True)
X = df_final.drop(['response'],axis = 1)
y = df_final['response']


X = sm.add_constant(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=5805, shuffle=True
)

random_forest_model = RandomForestClassifier(max_depth=20, random_state=5805, n_jobs=-1)
random_forest_model.fit(X_train,Y_train)
features = X.columns
importances = random_forest_model.feature_importances_
indices = np.argsort(importances)[::]
plt.figure(figsize = (12,12))
plt.title('Random Forest Feature Importances')
plt.barh(range(len(indices)),importances[indices],color = 'b', align = 'center')
plt.yticks(range(len(indices)),[features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.grid(True)
plt.show()


random_forest_model = RandomForestClassifier(max_depth=20, random_state=5805, n_jobs=-1)
random_forest_model.fit(X_train,Y_train)


threshold = 0.01
features_eliminated = [features[i] for i in indices if importances[i] < threshold]
features_selected = [features[i] for i in indices if importances[i] >= threshold]

print('Features SELECTED after Random Forest Analysis',features_selected);
print('Features ELIMINATED after Random Forest Analysis',features_eliminated);


# ---------------------------PCA-------------------------------------
pca = PCA(svd_solver='full', n_components=0.9, random_state=5805)
X_PCA = pca.fit_transform(X)
cumulative_variance_calculation = np.cumsum(pca.explained_variance_ratio_)
number_of_features_required = np.argmax(cumulative_variance_calculation >= 0.90) + 1

print('Cumulative variance',cumulative_variance_calculation)
print(f"Number of features needed to explain more than 90% of variance: {number_of_features_required}")
# Assuming X is a DataFrame with column names



plt.figure(figsize=(12, 12))
plt.plot(np.arange(1, len(cumulative_variance_calculation) + 1, 1),
         cumulative_variance_calculation)
plt.axhline(y=0.9,color='g',linestyle='--')
plt.axvline(x=number_of_features_required,color='r',linestyle='--')
plt.xticks((np.arange(1, len(cumulative_variance_calculation) + 1, 1)))
plt.xticks(rotation=25)
plt.title('Cumulative Explained Variance vs. Number of Features')
plt.xlabel('Number of features')
plt.ylabel('Cumulative explained variance')
plt.grid(True)
plt.show()

#------------------PCA Condition Number --------------------------------
cova_matrix = np.cov(X, rowvar=False)

condition_number = np.linalg.cond(X).round(3)


print(f"Condition Number before PCA is: {condition_number}")


condition_number_after_pca = np.linalg.cond(X_PCA).round(3)


print(f"Condition Number after PCA is: {condition_number_after_pca}")

print("--------------------------------\n")


#---------------------SVD------------------------------------------------
truncate_svd = TruncatedSVD(n_components=X_smote_resampled.shape[1] - 1)
svd_full = truncate_svd.fit_transform(X_smote_resampled)

cumulative_variance_ratio = np.cumsum(truncate_svd.explained_variance_ratio_)

features_above_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1

features_above_90_svd = TruncatedSVD(n_components=features_above_90)
features_reduced_after_svd = features_above_90_svd.fit_transform(X_smote_resampled)

print("Number of rows and columns before SVD: ", X_smote_resampled.shape)
print("Number of rows and columns after SVD: ", features_reduced_after_svd.shape)

print('Conditional  Number of data before SVD:', np.linalg.cond(X_smote_resampled).round(3))
print('Conditional Number of data after SVD:', np.linalg.cond(features_reduced_after_svd).round(3))

imp_features_names = X.columns[:features_above_90]
print("Important Features by SVD:")
print(imp_features_names)
print("--------------------------------------------\n")



#------------------Sample Pearson Correlation matrix-----------------------------------
plt.figure(figsize=(25, 25))
correlation_matrix = df_final.corr().round(1)
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu',xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns)
plt.title('Sample Pearson Correlation')
plt.show()


#------------------------Sample covariance matrix------------------------
plt.figure(figsize=(25, 25))
covariance_matrix = df_final.cov().round(1);
sns.heatmap(covariance_matrix, annot=True, cmap='RdBu',xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns)
plt.title('Sample Covariance Matrix')
plt.show()





#----------------------------------------------------VIF---------------------------------------------------------------
v_data= pd.DataFrame()
v_data["Feature"] = X.columns
v_data["VIF"] = [variance_inflation_factor(X, feature) for feature in range(X.shape[1])]

considered_threshold = 10

high_vif_features = v_data[v_data["VIF"] > considered_threshold]

print("Selected features after VIF analysis:")
num_important_features = len(high_vif_features)
print("Important Features number:", num_important_features)


important_features = high_vif_features["Feature"].tolist()
print("Names of Important Features:")
print(important_features)


























