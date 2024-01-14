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
from sklearn import tree
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


pd.set_option('display.max_columns', None)
#Reading the merged dataset

df_application_record= pd.read_csv("/Users/rashmikulkarni/VT-Fall/ML1/ML-Project/application_record.csv", low_memory=False)
print(df_application_record.head())

df_credit_record = pd.read_csv("/Users/rashmikulkarni/VT-Fall/ML1/ML-Project/credit_record.csv", low_memory=False)


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



df_application_record = df_application_record.drop_duplicates('ID')
df_credit_record = df_credit_record.drop_duplicates('ID')


df_final = df_application_record.merge(df_credit_record, on='ID', how='inner')


df_final.loc[(df_final['STATUS'] == 'X') | (df_final['STATUS'] == 'C') | (df_final['STATUS'] == '0'), 'response'] = 1
df_final.loc[(df_final['STATUS'] == '1') | (df_final['STATUS'] == '2') | (df_final['STATUS'] == '3') | (df_final['STATUS'] == '4') | (df_final['STATUS'] == '5'), 'response'] = 0





#--------Data cleaning----------------------------------------------------------------

df_final = df_final[df_final['CNT_CHILDREN'] >= 0]

df_final['CNT_FAM_MEMBERS'] = df_final['CNT_FAM_MEMBERS'].apply(lambda x: max(0, int(x)))
df_final['AMT_INCOME_TOTAL'] = df_final['AMT_INCOME_TOTAL'].apply(lambda x: max(0, float(x)))

df_bf_encoding = df_final

df_bf_encoding = df_bf_encoding.drop(columns=['ID','FLAG_MOBIL','CNT_CHILDREN','DAYS_BIRTH'])

#Applying apriori algorithm


# # Assuming df_final is your DataFrame
df_bf_encoding = df_bf_encoding.select_dtypes(exclude='number')
# print("columns",df_bf_encoding.columns)
df_bf_encoding = pd.get_dummies(df_bf_encoding, columns = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE','STATUS'], drop_first=True)
#
#
print('---------- Results of Association Rule Mining ----------')
occurring_together = apriori(df_bf_encoding, min_support=0.2, use_colnames=True, verbose=1)
print(occurring_together)

formed_rules = association_rules(occurring_together, metric='confidence', min_threshold=0.7)
formed_rules = formed_rules.sort_values(['confidence','lift'], ascending=[False,False])
print(formed_rules.to_string())



#-----------------------------------------------------------K means--------------------------------------------------------

#------------One hot encoding--------------------------------
df_final = pd.get_dummies(df_final, columns = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'], drop_first=True)


#--------Removing columns with no information like ID--------------------------------
df_final = df_final.drop(columns=['ID','STATUS','CNT_CHILDREN','DAYS_BIRTH','FLAG_MOBIL'])#RESPONSE BASED ON STATUS
df_final = df_final[: : 4]

# Applying SMOTE
df_final = df_final.reset_index()
X = df_final.drop(columns=['response'], axis=1)

y = df_final['response']


smote = SMOTE(random_state=5805)
X_smote_resampled, y_smote_resampled = smote.fit_resample(X, y)

df_final = pd.concat([pd.DataFrame(X_smote_resampled, columns=X.columns), pd.DataFrame({'response': y_smote_resampled})], axis=1)

#---------- Standardisation--------------------------------------------------------------------------------------------
standardize_features = ['AMT_INCOME_TOTAL','DAYS_EMPLOYED','CNT_FAM_MEMBERS','MONTHS_BALANCE']


def standardize(df):
    standardized_df = round((df - df.mean()) / df.std(), 2)
    return standardized_df



df_final[standardize_features] = standardize(df_final[standardize_features])

# ========================================================
# # Elbow Method Within cluster Sum of Squared Errors (WSS)
# #========================================================

def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0
        # calculate square of Euclidean distance of each point
        # from its cluster center and add to current WSS
        for i in range(len(df_final)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)

    return sse

k = 10
sse = calculate_WSS(df_final.values, k)
plt.figure()
plt.plot(np.arange(1, k + 1, 1), np.array(sse), label='WSS')
plt.xticks(np.arange(1, k + 1, 1))
plt.grid()
plt.xlabel('k')
plt.ylabel('WSS')
plt.title('K selection in k-means Elbow Algorithm')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
#elbow at 2. K=2 optimum

#========================================================
# # Silhouette Method for selection of K
# #========================================================
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
sil = []
kmax = 15

for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters=k).fit(X)
    labels = kmeans.labels_
    sil.append(silhouette_score(X, labels, metric='euclidean'))
plt.figure()
plt.plot(np.arange(2, k + 1, 1), sil, 'bx-',label='Silhouette score')
plt.xticks(np.arange(2,k+1,1))
plt.grid()
plt.xlabel('K')
plt.ylabel('Silhouette score')
plt.title('Silhouette Method')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


#========================================================code for kmeans elbow and silhoutte taken from prof reza's lecture files






