import pandas as pd
import numpy as np
import seaborn as seaborn
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
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
    classification_report, ConfusionMatrixDisplay, mean_squared_error, auc
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn import tree
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import StackingClassifier
import warnings
warnings.filterwarnings("ignore")

np.random.seed(5805)


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


df_final.loc[(df_final['STATUS'] == 'X') | (df_final['STATUS'] == 'C') | (df_final['STATUS'] == '0'), 'response'] = 1
df_final.loc[(df_final['STATUS'] == '1') | (df_final['STATUS'] == '2') | (df_final['STATUS'] == '3') | (df_final['STATUS'] == '4') | (df_final['STATUS'] == '5'), 'response'] = 0



#Data cleaning

df_final = df_final[df_final['CNT_CHILDREN'] >= 0]
df_final['CNT_FAM_MEMBERS'] = df_final['CNT_FAM_MEMBERS'].apply(lambda x: max(0, int(x)))
df_final['AMT_INCOME_TOTAL'] = df_final['AMT_INCOME_TOTAL'].apply(lambda x: max(0, float(x)))

#------------One hot encoding--------------------------------
df_final = pd.get_dummies(df_final, columns = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'], drop_first=True)

#--------Removing columns with no information like ID--------------------------------
df_final = df_final.drop(columns=['ID','STATUS','FLAG_MOBIL','CNT_CHILDREN','DAYS_BIRTH'])
# Downsampling rows
df_final = df_final[: : 4]
print("After downsampling",df_final.shape[0])

# Applying SMOTE
df_final = df_final.reset_index()
X = df_final.drop(columns=['response'], axis=1)
y = df_final['response']


smote = SMOTE(random_state=5805)
X_smote_resampled, y_smote_resampled = smote.fit_resample(X, y)

df_final = pd.concat([pd.DataFrame(X_smote_resampled, columns=X.columns), pd.DataFrame({'response': y_smote_resampled})], axis=1)

print("After smote",df_final.shape[0])




#---------- Standardisation--------------------------------------------------------------------------------------------
standardize_features = ['AMT_INCOME_TOTAL','DAYS_EMPLOYED','CNT_FAM_MEMBERS','MONTHS_BALANCE']


def standardize(df):
    standardized_df = round((df - df.mean()) / df.std(), 2)
    return standardized_df


df_final[standardize_features] = standardize(df_final[standardize_features])
#-----------------------------------------------------------------------------------------


def kfold_cross_validation(model, X, y, k=2, verbose=True, random_state=None):
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    scores = {
        "accuracy": []
    }

    fold = 0
    for train_index, test_index in kf.split(X, y):
        fold += 1
        X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
        y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]

        try:
            model.fit(X_train_kf, y_train_kf)
            y_pred_kf = model.predict(X_test_kf)

            scores["accuracy"].append(accuracy_score(y_test_kf, y_pred_kf))


            if verbose:
                print(f'Fold {fold} - Accuracy: {scores["accuracy"][-1]:.3f}')
        except NotFittedError as e:
            print(f'Error in fold {fold}: {e}')

    avg_scores = {
        "average_accuracy": np.mean(scores["accuracy"])
    }


    return avg_scores


#-------------------------------------------------Decision Tree--------------------------------------------------------
print("DF size before classification",df_final.shape[0])

X_dt = df_final[ ['OCCUPATION_TYPE_High skill tech staff', 'OCCUPATION_TYPE_Medicine staff', 'OCCUPATION_TYPE_Security staff', 'NAME_HOUSING_TYPE_With parents', 'FLAG_EMAIL',
                  'NAME_HOUSING_TYPE_House / apartment', 'NAME_FAMILY_STATUS_Separated', 'NAME_EDUCATION_TYPE_Incomplete higher',
                  'NAME_INCOME_TYPE_State servant', 'OCCUPATION_TYPE_Drivers', 'CODE_GENDER_M',
                  'OCCUPATION_TYPE_Managers', 'NAME_FAMILY_STATUS_Single / not married',
                  'NAME_FAMILY_STATUS_Married', 'NAME_INCOME_TYPE_Working', 'FLAG_OWN_CAR_Y',
                  'DAYS_EMPLOYED', 'OCCUPATION_TYPE_Core staff', 'FLAG_OWN_REALTY_Y', 'CNT_FAM_MEMBERS',
                  'FLAG_WORK_PHONE', 'FLAG_PHONE', 'OCCUPATION_TYPE_Sales staff',
                  'NAME_EDUCATION_TYPE_Secondary / secondary special', 'NAME_EDUCATION_TYPE_Higher education',
                  'OCCUPATION_TYPE_Laborers', 'AMT_INCOME_TOTAL']
]
y_dt = df_final['response']

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=0.2, random_state=5805)

# Create a decision tree classifier
decision_tree_model = DecisionTreeClassifier()

decision_tree_model.fit(X_train, y_train)

y_pred_dt = decision_tree_model.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Kfold cross validation Decision Tree basic--------------------------------")
kfold_cross_validation(decision_tree_model,X_train,y_train)
print("Accuracy of Decision Tree Classifier before grid:", accuracy_dt.round(3))

# Confusion matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_dt, annot=True, cmap='PuRd', fmt='d')
plt.title('Confusion Matrix for Decision Tree Classifier before grid')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

precision_dt = precision_score(y_test, y_pred_dt )
recall_dt = recall_score(y_test, y_pred_dt )
f_score_dt = f1_score(y_test, y_pred_dt )

# Display precision, recall, and F-score
print("Precision for Decision Tree before prepruning:", precision_dt.round(3))
print("Recall for Decision Tree before prepruning:", recall_dt.round(3))
print("F-score for Decision Tree before prepruning:", f_score_dt.round(3))
print("Specificity for Decision Tree before prepruning:", (cm_dt[0,0] / (cm_dt[0,0] + cm_dt[0,1])).round(3))
print("Confusion matrix :", cm_dt)
print("----------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_dt)
roc_auc_dt = auc(fpr, tpr)

# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_dt)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve for Decision Tree  before prepruning')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#Prepruning(grid search)
# Define the hyperparameters to tune
to_tune_hp = {
    'criterion': ['gini', 'entropy','log-loss'],
    'splitter': ['best', 'random'],
    'max_depth': [2, 8, 10],
    'min_samples_split': [2, 4, 6],
    'max_features': ['auto', 'sqrt'],
    'ccp_alpha': np.linspace(0, 0.05, 10)
}

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(decision_tree_model, to_tune_hp, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params_dt = grid_search.best_params_
print("Best hyperparameters:", best_params_dt)

print('The Final tree is:', grid_search.best_estimator_)

gs_best_estimator_params = grid_search.best_estimator_
gs_best_estimator_params.fit(X_train,y_train)

y_pred_gridsearch = gs_best_estimator_params.predict(X_test)

# accuracy_dt = accuracy_score(y_test, y_pred_gridsearch).round(3)
# print('Accuracy of Decision Tree Classifier after pre-pruning : ',accuracy_dt)


plt.figure(figsize=(12, 8))
plt.title('Decision Tree After Pre-Pruning')
plt.xlabel('Feature')
plt.ylabel('Target Class')
tree.plot_tree(grid_search.best_estimator_, filled=True, class_names=["blue", "crimson"], rounded=True)
plt.show()


# Train the decision tree classifier with the best hyperparameters
best_dtree = DecisionTreeClassifier(**best_params_dt)
best_dtree.fit(X_train, y_train)

# Make predictions
predictions = best_dtree.predict(X_test)

# Evaluate the model's performance
accuracy_dt_prepruned = accuracy_score(y_test, predictions)

print("Kfold cross validation Decision Tree after pre-pruning--------------------------------")
kfold_cross_validation(best_dtree,X_train,y_train)
print("Accuracy of Decision Tree Classifier after pre-pruning with best hyper parameters:", accuracy_dt_prepruned.round(3))

# Confusion matrix
cm_pp = confusion_matrix(y_test, predictions)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_pp, annot=True, cmap='BuGn', fmt='d')
plt.title('Confusion Matrix for Decision Tree Classifier after pre-pruning')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

precision_dt_pp = precision_score(y_test, predictions )
recall_dt_pp = recall_score(y_test, predictions )
f_score_dt_pp = f1_score(y_test, predictions )

# Display precision, recall, and F-score
print("Precision for Decision Tree after prepruning:", precision_dt_pp.round(3))
print("Recall for Decision Tree after prepruning:", recall_dt_pp.round(3))
print("F-score for Decision Tree after prepruning:", f_score_dt_pp.round(3))
print("Specificity for Decision Tree after prepruning:", (cm_pp[0,0] / (cm_pp[0,0] + cm_pp[0,1])).round(3))
print("Confusion matrix :", cm_pp)
print("----------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_dt)
roc_auc_dt_pp = auc(fpr, tpr)

# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_dt_pp)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve for Decision Tree after prepruning')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#----------------------------------------------Post Pruning ccp-alpha---------------------------------------------
path_opt_alpha = best_dtree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path_opt_alpha.ccp_alphas, path_opt_alpha.impurities


train_acc, test_acc = [], []


for ccp_alpha in ccp_alphas:
    pruned_model = DecisionTreeClassifier(random_state=5805, ccp_alpha=ccp_alpha)
    pruned_model.fit(X_train, y_train)
    y_pred_alpha = pruned_model.predict(X_test)
    y_pred_beta = pruned_model.predict(X_train)
    train_acc.append(accuracy_score(y_train, y_pred_beta))
    test_acc.append(accuracy_score(y_test, y_pred_alpha))

opt_ccp_alpha = ccp_alphas[np.argmax(test_acc)]


pruned_model = DecisionTreeClassifier(random_state=5805, ccp_alpha=opt_ccp_alpha)
pruned_model.fit(X_train, y_train)
tree.plot_tree(pruned_model, filled=True, class_names=["blue", "crimson"], rounded=True,)


fig, ax = plt.subplots()
ax.plot(ccp_alphas, train_acc, marker='o', drawstyle="steps-post", label='Train Accuracy')
ax.plot(ccp_alphas, test_acc, marker='o', drawstyle="steps-post", label='Test Accuracy')
ax.set_xlabel("Effective alpha")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs effective alpha")
ax.legend()
plt.show()


y_pred_dt_ccp = pruned_model.predict(X_test)
print('Accuracy after post-pruning the tree:', accuracy_score(y_test, y_pred_dt_ccp).round(2))

print('Optimum alpha after rounding off to decimal places is:', opt_ccp_alpha.round(2))
print('Optimum alpha without rounding off is:',opt_ccp_alpha)

accuracy_dt_ccp = accuracy_score(y_test, y_pred_dt_ccp)
print("Accuracy of Decision Tree after post-pruning:", accuracy_dt_ccp.round(3))
print("Kfold cross validation for decision Tree post-pruning--------------------------------")
kfold_cross_validation(pruned_model,X_train,y_train)

# Confusion matrix
cm_ppp = confusion_matrix(y_test, y_pred_dt_ccp)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_ppp, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix for Decision Tree after post pruning')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

precision_dt_ccp = precision_score(y_test, y_pred_dt_ccp )
recall_dt_ccp = recall_score(y_test, y_pred_dt_ccp )
f_score_dt_ccp = f1_score(y_test, y_pred_dt_ccp )

# Display precision, recall, and F-score
print("Precision for Decision Tree post pruning:", precision_dt_ccp.round(3))
print("Recall for Decision Tree post pruning:", recall_dt_ccp.round(3))
print("F-score for Decision Tree post pruning:", f_score_dt_ccp.round(3))
print("Specificity for Decision Tree post pruning:", (cm_ppp[0,0] / (cm_ppp[0,0] + cm_ppp[0,1])).round(3))
print("Confusion matrix :", cm_ppp)
print("----------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_dt_ccp)
roc_auc_dt_ccp = auc(fpr, tpr)
roc_auc_dt_ccp = roc_auc_score(y_test, y_pred_dt_ccp)


# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_dt_ccp)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve for Decision Tree after post pruning')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


#-------------------------Logistic Regression--------------------------------------------------------------

X_lr = df_final[ ['OCCUPATION_TYPE_High skill tech staff', 'OCCUPATION_TYPE_Medicine staff', 'OCCUPATION_TYPE_Security staff', 'NAME_HOUSING_TYPE_With parents', 'FLAG_EMAIL',
                  'NAME_HOUSING_TYPE_House / apartment', 'NAME_FAMILY_STATUS_Separated', 'NAME_EDUCATION_TYPE_Incomplete higher',
                  'NAME_INCOME_TYPE_State servant', 'OCCUPATION_TYPE_Drivers', 'CODE_GENDER_M',
                  'OCCUPATION_TYPE_Managers', 'NAME_FAMILY_STATUS_Single / not married',
                  'NAME_FAMILY_STATUS_Married', 'NAME_INCOME_TYPE_Working', 'FLAG_OWN_CAR_Y',
                  'DAYS_EMPLOYED', 'OCCUPATION_TYPE_Core staff', 'FLAG_OWN_REALTY_Y', 'CNT_FAM_MEMBERS',
                  'FLAG_WORK_PHONE', 'FLAG_PHONE', 'OCCUPATION_TYPE_Sales staff',
                  'NAME_EDUCATION_TYPE_Secondary / secondary special', 'NAME_EDUCATION_TYPE_Higher education',
                  'OCCUPATION_TYPE_Laborers', 'AMT_INCOME_TOTAL']
]
y_lr = df_final['response']

logistic_regression_model = LogisticRegression(max_iter=100,random_state=5805)
logistic_regression_model.fit(X_train, y_train)
y_predicted_lr = logistic_regression_model.predict(X_test)

# Evaluate the model's performance
accuracy_lr = accuracy_score(y_test, y_predicted_lr)
print("Accuracy of Logistic Regression before grid search:", accuracy_lr.round(3))
print("Kfold cross validation logistic regression-----------------------------------------------")
kfold_cross_validation(logistic_regression_model,X_train,y_train)


# Confusion matrix
cm_lr = confusion_matrix(y_test, y_predicted_lr)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_lr, annot=True, cmap='YlOrRd', fmt='d')
plt.title('Confusion Matrix for Logistic Regression before grid search')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

precision_lr = precision_score(y_test, y_predicted_lr )
recall_lr = recall_score(y_test, y_predicted_lr )
f_score_lr = f1_score(y_test, y_predicted_lr )

# Display precision, recall, and F-score
print("Precision for Logistic Regression before grid search:", precision_lr.round(3))
print("Recall for Logistic Regression before grid search:", recall_lr.round(3))
print("F-score for Logistic Regression before grid search:", f_score_lr.round(3))
print("Specificity for Logistic Regression before grid search:", (cm_lr[0,0] / (cm_lr[0,0] + cm_lr[0,1])).round(3))
print("Confusion matrix :", cm_lr)
print("--------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_predicted_lr)
roc_auc_lr = auc(fpr, tpr)
auc_lr = roc_auc_score(y_test, y_predicted_lr)


# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_lr)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve for Logistic Regression before grid search')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Grid search for logistic regression

params_lr = [{'C': [0.01, 1, 10,100], 'solver': ['liblinear', 'newton-cg'],
              'max_iter': [500, 2000]}]



grid_search_lr = GridSearchCV(estimator=logistic_regression_model, param_grid=params_lr, cv=5, scoring='accuracy')

# Fit the model
grid_search_lr.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search_lr.best_params_

# Use the best parameters to train the final model
final_model_lr = LogisticRegression(**best_params)
final_model_lr.fit(X_train, y_train)

# Predict on the test set
y_pred_lr_g = final_model_lr.predict(X_test)

# Evaluate the accuracy
accuracy_lr_g = accuracy_score(y_test, y_pred_lr_g)
print("Accuracy of Logistic Regression after grid search:", accuracy_lr_g.round(3))
print("Kfold cross validation logistic regression after grid search-----------------------------------------------")
kfold_cross_validation(final_model_lr,X_train,y_train)

# Confusion matrix
cm_lr_g = confusion_matrix(y_test, y_pred_lr_g)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_lr, annot=True, cmap='YlOrRd', fmt='d')
plt.title('Confusion Matrix for Logistic Regression after grid search')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

precision_lr_g = precision_score(y_test, y_pred_lr_g )
recall_lr_g = recall_score(y_test, y_pred_lr_g )
f_score_lr_g = f1_score(y_test, y_pred_lr_g )

# Display precision, recall, and F-score
print("Precision for Logistic Regression after grid search:", precision_lr_g.round(3))
print("Recall for Logistic Regression after grid search:", recall_lr_g.round(3))
print("F-score for Logistic Regression after grid search:", f_score_lr_g.round(3))
print("Specificity for Logistic Regression after grid search:", (cm_lr_g[0,0] / (cm_lr_g[0,0] + cm_lr_g[0,1])).round(3))
print("Confusion matrix :", cm_lr_g)
print("--------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_lr_g)
roc_auc_lr_g = auc(fpr, tpr)
auc_lr = roc_auc_score(y_test, y_pred_lr_g)


# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_lr_g)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve for Logistic Regression after grid search')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
#-------------------------KNN------------------------------------
#df = pd.DataFrame(dataset.data, columns= dataset.feature_names)

X_knn = df_final[ ['OCCUPATION_TYPE_High skill tech staff', 'OCCUPATION_TYPE_Medicine staff', 'OCCUPATION_TYPE_Security staff', 'NAME_HOUSING_TYPE_With parents', 'FLAG_EMAIL',
                  'NAME_HOUSING_TYPE_House / apartment', 'NAME_FAMILY_STATUS_Separated', 'NAME_EDUCATION_TYPE_Incomplete higher',
                  'NAME_INCOME_TYPE_State servant', 'OCCUPATION_TYPE_Drivers', 'CODE_GENDER_M',
                  'OCCUPATION_TYPE_Managers', 'NAME_FAMILY_STATUS_Single / not married',
                  'NAME_FAMILY_STATUS_Married', 'NAME_INCOME_TYPE_Working', 'FLAG_OWN_CAR_Y',
                  'DAYS_EMPLOYED', 'OCCUPATION_TYPE_Core staff', 'FLAG_OWN_REALTY_Y', 'CNT_FAM_MEMBERS',
                  'FLAG_WORK_PHONE', 'FLAG_PHONE', 'OCCUPATION_TYPE_Sales staff',
                  'NAME_EDUCATION_TYPE_Secondary / secondary special', 'NAME_EDUCATION_TYPE_Higher education',
                  'OCCUPATION_TYPE_Laborers', 'AMT_INCOME_TOTAL']
]
y_knn = df_final['response']
# #=============================================
# # Fitting KNN Classifier to the Training set
# #=============================================
x_train, x_test, y_train, y_test = train_test_split(X_knn, y_knn,
test_size = 0.20,
random_state = 5805)
model_knn = KNeighborsClassifier(n_neighbors=2, metric='minkowski', p=2)
model_knn.fit(x_train, y_train)
# Predicting the results
y_pred_knn= model_knn.predict(x_test)

# Evaluate the model's performance
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy of KNN before grid search:", accuracy_knn.round(3))
print("Kfold cross validation for KNN-----------------------------------------------")
kfold_cross_validation(model_knn,X_train,y_train)


# Confusion matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_knn, annot=True, cmap='PuBu', fmt='d')
plt.title('Confusion Matrix for KNN before grid search')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

precision_knn = precision_score(y_test, y_pred_knn )
recall_knn = recall_score(y_test, y_pred_knn )
f_score_knn = f1_score(y_test, y_pred_knn )

# Display precision, recall, and F-score
print("Precision for KNN before grid search:", precision_knn.round(3))
print("Recall for KNN before grid search:", precision_knn.round(3))
print("F-score for KNN before grid search:", precision_knn.round(3))
print("Specificity for KNN before grid search:", (cm_knn[0,0] / (cm_knn[0,0] + cm_knn[0,1])).round(3))
print("Confusion matrix :", cm_knn)
print("--------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_knn)
roc_auc_knn = auc(fpr, tpr)
auc_knn = roc_auc_score(y_test, y_pred_knn)


# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_knn)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve for KNN before grid search')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# #=========================
# # Grid Search for best K
# #==========================
knn_model = KNeighborsClassifier(metric ='euclidean')
knn_params = list(range(2, 5))
params_knn = dict(n_neighbors=knn_params)
grid = GridSearchCV(knn_model, params_knn, cv=5, scoring='accuracy',
return_train_score=True, verbose=1)

# fitting the model for grid search
grid_search_knn = grid.fit(x_train, y_train)
print(f'The best K value is : {grid_search_knn.best_params_}')

# viii) Checking Accuracy on Test Data
knn_g = KNeighborsClassifier(n_neighbors = 3)
knn_g.fit(x_train, y_train)
y_pred_knn_g=knn_g.predict(x_test)
accuracy_knn_g=accuracy_score(y_test,y_pred_knn_g)*100

accuracy_knn_g = accuracy_score(y_test, y_pred_knn_g)
print("Accuracy of KNN after grid search:", accuracy_knn_g.round(3))
print("Kfold cross validation for KNN-----------------------------------------------")
kfold_cross_validation(knn_g,X_train,y_train)


# Confusion matrix
cm_knn_g = confusion_matrix(y_test, y_pred_knn_g)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_knn, annot=True, cmap='PuBu', fmt='d')
plt.title('Confusion Matrix for KNN after grid search')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

precision_knn_g = precision_score(y_test, y_pred_knn_g )
recall_knn_g = recall_score(y_test, y_pred_knn_g )
f_score_knn_g= f1_score(y_test, y_pred_knn_g )

# Display precision, recall, and F-score
print("Precision for KNN after grid search:", precision_knn_g.round(3))
print("Recall for KNN after grid search:", recall_knn_g.round(3))
print("F-score for KNN after grid search:", f_score_knn_g.round(3))
print("Specificity for KNN after grid search:", (cm_knn_g[0,0] / (cm_knn_g[0,0] + cm_knn_g[0,1])).round(3))
print("Confusion matrix :", cm_knn_g)
print("--------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_knn_g)
roc_auc_knn_g = auc(fpr, tpr)
auc_knn = roc_auc_score(y_test, y_pred_knn_g)


# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_knn_g)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve for KNN after grid search')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# #-------------------------------------------KNN WITH ELBOW METHOD--------------------------------------------------
list_k = list(range(2, 10))
error_rate = []

for k in list_k:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    error_rate.append(np.mean(y_pred_knn != y_test))

plt.figure(figsize=(12,8))
plt.plot(range(2,10),error_rate,color='blue', marker='o',
 markerfacecolor='red', markersize=12)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.grid(True)
plt.show()
# #-------------------------------------------SVM CLASSIFIER--------------------------------------------------
X_svm = df_final[ ['OCCUPATION_TYPE_High skill tech staff', 'OCCUPATION_TYPE_Medicine staff', 'OCCUPATION_TYPE_Security staff', 'NAME_HOUSING_TYPE_With parents', 'FLAG_EMAIL',
                  'NAME_HOUSING_TYPE_House / apartment', 'NAME_FAMILY_STATUS_Separated', 'NAME_EDUCATION_TYPE_Incomplete higher',
                  'NAME_INCOME_TYPE_State servant', 'OCCUPATION_TYPE_Drivers', 'CODE_GENDER_M',
                  'OCCUPATION_TYPE_Managers', 'NAME_FAMILY_STATUS_Single / not married',
                  'NAME_FAMILY_STATUS_Married', 'NAME_INCOME_TYPE_Working', 'FLAG_OWN_CAR_Y',
                  'DAYS_EMPLOYED', 'OCCUPATION_TYPE_Core staff', 'FLAG_OWN_REALTY_Y', 'CNT_FAM_MEMBERS',
                  'FLAG_WORK_PHONE', 'FLAG_PHONE', 'OCCUPATION_TYPE_Sales staff',
                  'NAME_EDUCATION_TYPE_Secondary / secondary special', 'NAME_EDUCATION_TYPE_Higher education',
                  'OCCUPATION_TYPE_Laborers', 'AMT_INCOME_TOTAL']
]
y_svm = df_final['response']

X_train, X_test, y_train, y_test = train_test_split(X_svm, y_svm, test_size=0.2, random_state=5805)
# Create a linear SVM model
linear_svm_model = svm.SVC(kernel='linear')
linear_svm_model.fit(X_train, y_train)
y_pred_svm = linear_svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Accuracy of SVM before grid search:", accuracy_svm.round(3))
print("Kfold cross validation for SVM-----------------------------------------------")
kfold_cross_validation(linear_svm_model,X_train,y_train)


# Confusion matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_svm, annot=True, cmap='GnBu', fmt='d')
plt.title('Confusion Matrix for SVM before grid search')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

precision_svm = precision_score(y_test, y_pred_svm )
recall_svm = recall_score(y_test, y_pred_svm )
f_score_svm= f1_score(y_test, y_pred_svm )

# Display precision, recall, and F-score
print("Precision for SVM before grid search:", precision_svm.round(3))
print("Recall for SVM before grid search:", recall_svm.round(3))
print("F-score for SVM before grid search:", f_score_svm.round(3))
print("Specificity for SVM before grid search:", (cm_svm[0,0] / (cm_svm[0,0] + cm_svm[0,1])).round(3))
print("Confusion matrix :", cm_svm)
print("----------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_svm)
roc_auc_svm = auc(fpr, tpr)
auc_svm = roc_auc_score(y_test, y_pred_svm)


# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_svm)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve for SVM before grid search')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# --------------------------------------Grid search for polynomial and radial base kernel------------------------------------------------
param_grid = {'C': [0.1,10], 'gamma': [1, 0.001], 'kernel': ['poly', 'rbf']}

# Create the SVM model
svm_model = svm.SVC()

# Perform the grid search
grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)


print("Best parameters for SVM after grid search: ", grid_search.best_params_)

#----------------------Training the svm with parameters given by grid search------------------------------------------------

# Create a new SVM model with the best parameters
svm_model_after_grid = svm.SVC(C=10, kernel='rbf', gamma=1)

# Train the model with the best parameters
svm_model_after_grid.fit(X_train, y_train)

y_pred_svm_grid = svm_model_after_grid.predict(X_test)
accuracy_svm_grid = accuracy_score(y_test, y_pred_svm_grid)
print("Accuracy of SVM after grid search:", accuracy_svm_grid.round(3))
print("Kfold cross validation for SVM-----------------------------------------------")
kfold_cross_validation(svm_model_after_grid,X_train,y_train)


# Confusion matrix
cm_svm_grid = confusion_matrix(y_test, y_pred_svm_grid)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_svm_grid, annot=True, cmap='GnBu', fmt='d')
plt.title('Confusion Matrix for SVM after grid search')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

precision_svm_grid = precision_score(y_test, y_pred_svm_grid )
recall_svm_grid = recall_score(y_test, y_pred_svm_grid )
f_score_svm_grid= f1_score(y_test, y_pred_svm_grid )

# Display precision, recall, and F-score
print("Precision for SVM after grid search:", precision_svm_grid.round(3))
print("Recall for SVM after grid search:", recall_svm_grid.round(3))
print("F-score for SVM after grid search:", f_score_svm_grid.round(3))
print("Specificity for SVM after grid search:", (cm_svm_grid[0,0] / (cm_svm_grid[0,0] + cm_svm_grid[0,1])).round(3))
print("Confusion matrix :", cm_svm_grid)
print("----------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_svm_grid)
roc_auc_svm_grid = auc(fpr, tpr)
auc_svm = roc_auc_score(y_test, y_pred_svm_grid)


# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_svm_grid)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve for SVM after grid search')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()




#--------------------------------------Naive Bayes----------------------------------------------------------------

X_nb = df_final[ ['OCCUPATION_TYPE_High skill tech staff', 'OCCUPATION_TYPE_Medicine staff', 'OCCUPATION_TYPE_Security staff', 'NAME_HOUSING_TYPE_With parents', 'FLAG_EMAIL',
                  'NAME_HOUSING_TYPE_House / apartment', 'NAME_FAMILY_STATUS_Separated', 'NAME_EDUCATION_TYPE_Incomplete higher',
                  'NAME_INCOME_TYPE_State servant', 'OCCUPATION_TYPE_Drivers', 'CODE_GENDER_M',
                  'OCCUPATION_TYPE_Managers', 'NAME_FAMILY_STATUS_Single / not married',
                  'NAME_FAMILY_STATUS_Married', 'NAME_INCOME_TYPE_Working', 'FLAG_OWN_CAR_Y',
                  'DAYS_EMPLOYED', 'OCCUPATION_TYPE_Core staff', 'FLAG_OWN_REALTY_Y', 'CNT_FAM_MEMBERS',
                  'FLAG_WORK_PHONE', 'FLAG_PHONE', 'OCCUPATION_TYPE_Sales staff',
                  'NAME_EDUCATION_TYPE_Secondary / secondary special', 'NAME_EDUCATION_TYPE_Higher education',
                  'OCCUPATION_TYPE_Laborers', 'AMT_INCOME_TOTAL']
]
y_nb = df_final['response']

X_train, X_test, y_train, y_test = train_test_split(X_nb, y_nb, test_size=0.2, random_state=5805)

nb_model = GaussianNB()

nb_model.fit(X_train, y_train)


y_pred_nb = nb_model.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("KFold validation-------------------------------- Naive Bayes")
kfold_cross_validation(nb_model,X_train,y_train)
print("Accuracy of Gaussian Naive Bayes classifier before grid:", accuracy_nb.round(3))

# Confusion matrix
cm_nb = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_nb, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix for Naive Bayes')
plt.show()

precision_nb = precision_score(y_test, y_pred_nb )
recall_nb = recall_score(y_test, y_pred_nb )
f_score_nb = f1_score(y_test, y_pred_nb )

# Display precision, recall, and F-score
print("Precision for Naive Bayes before grid search:", precision_nb.round(3))
print("Recall for Naive Bayes before grid search:", recall_nb.round(3))
print("F-score for Naive Bayes before grid search:", f_score_nb.round(3))
print("Specificity for Naive Bayes before grid search:", (cm_nb[0,0] / (cm_nb[0,0] + cm_nb[0,1])).round(3))
print("Confusion matrix :", cm_nb)
print("----------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_nb)
roc_auc_nb = auc(fpr, tpr)
# auc_nb = roc_auc_score(y_test, y_pred_nb)

# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_nb)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Naive Bayes before grid search')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#Using alpha values to tune NB---------------------------------

params = [{'var_smoothing':[1e-9, 1e-5, 1e-3]}]

grid_search_nb = GridSearchCV(estimator=nb_model, param_grid=params, cv=5, scoring='accuracy')

# Fit the model
grid_search_nb.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search_nb.best_params_

# Use the best parameters to train the final model
final_model = GaussianNB(**best_params)
final_model.fit(X_train, y_train)

# Predict on the test set
y_pred_nb_g = final_model.predict(X_test)

# Evaluate the accuracy
accuracy_nb_g = accuracy_score(y_test, y_pred_nb_g)
print("Accuracy of Gaussian Naive Bayes classifier after grid:", accuracy_nb_g.round(3))
kfold_cross_validation(final_model,X_train,y_train)

# Confusion matrix
cm_nb_g = confusion_matrix(y_test, y_pred_nb_g)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_nb_g, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix for Naive Bayes after grid search')
plt.show()

precision_nb_g = precision_score(y_test, y_pred_nb_g )
recall_nb_g = recall_score(y_test, y_pred_nb_g )
f_score_nb_g = f1_score(y_test, y_pred_nb_g )

# Display precision, recall, and F-score
print("Precision for Naive Bayes after grid search:", precision_nb_g.round(3))
print("Recall for Naive Bayes after grid search:", recall_nb_g.round(3))
print("F-score for Naive Bayes after grid search:", f_score_nb_g.round(3))
print("Specificity for Naive Bayes after grid search:", (cm_nb_g[0,0] / (cm_nb_g[0,0] + cm_nb_g[0,1])).round(3))
print("Confusion matrix :", cm_nb_g)
print("----------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_nb_g)
auc_nb = roc_auc_score(y_test, y_pred_nb_g)


# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_nb)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Naive Bayes after grid search')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# #--------------------------------------Random Forest Classifier------------------------------------------------
#
X_rf = df_final[ ['OCCUPATION_TYPE_High skill tech staff', 'OCCUPATION_TYPE_Medicine staff', 'OCCUPATION_TYPE_Security staff', 'NAME_HOUSING_TYPE_With parents', 'FLAG_EMAIL',
                  'NAME_HOUSING_TYPE_House / apartment', 'NAME_FAMILY_STATUS_Separated', 'NAME_EDUCATION_TYPE_Incomplete higher',
                  'NAME_INCOME_TYPE_State servant', 'OCCUPATION_TYPE_Drivers', 'CODE_GENDER_M',
                  'OCCUPATION_TYPE_Managers', 'NAME_FAMILY_STATUS_Single / not married',
                  'NAME_FAMILY_STATUS_Married', 'NAME_INCOME_TYPE_Working', 'FLAG_OWN_CAR_Y',
                  'DAYS_EMPLOYED', 'OCCUPATION_TYPE_Core staff', 'FLAG_OWN_REALTY_Y', 'CNT_FAM_MEMBERS',
                  'FLAG_WORK_PHONE', 'FLAG_PHONE', 'OCCUPATION_TYPE_Sales staff',
                  'NAME_EDUCATION_TYPE_Secondary / secondary special', 'NAME_EDUCATION_TYPE_Higher education',
                  'OCCUPATION_TYPE_Laborers', 'AMT_INCOME_TOTAL']
]
y_rf = df_final['response']

X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=5805)

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=5805)

# Train the classifier
random_forest_model.fit(X_train, y_train)

# Predict the response for test dataset
y_pred_rf = random_forest_model.predict(X_test)

# Calculate the accuracy of the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy of Random Forest Classifier:", accuracy_rf.round(3))
print("Kfold cross validation Random Forest-----------------------------------------------")
kfold_cross_validation(random_forest_model,X_train,y_train)

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_rf, annot=True, cmap='summer', fmt='d')
plt.title('Confusion Matrix for Random Forest before grid search')
plt.show()

precision_rf= precision_score(y_test, y_pred_rf )
recall_rf = recall_score(y_test, y_pred_rf )
f_score_rf = f1_score(y_test, y_pred_rf )

# Display precision, recall, and F-score
print("Precision for Random Forest before grid search:", precision_rf.round(3))
print("Recall for Random Forest before grid search:", recall_rf.round(3))
print("F-score for Random Forest before grid search:", f_score_rf.round(3))
print("Specificity for Random Forest before grid search:", (cm_rf[0,0] / (cm_rf[0,0] + cm_rf[0,1])).round(3))
print("Confusion matrix :", cm_rf)
print("----------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)


# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest before grid search')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()




#------------------------------------Random Forest Grid Search----------------------------------------------------

rf_params={'max_depth':[3,5],
              'n_estimators':[100,200],
              'max_features':[1,3],
              'min_samples_leaf':[1,3],
              'min_samples_split':[1,3]}

grid = GridSearchCV(random_forest_model,param_grid=rf_params,cv=3,scoring='accuracy')
model_grid_rf_g = grid.fit(X,y)
best_params_rf = model_grid_rf_g.best_params_
final_model_rf_grid = RandomForestClassifier(**best_params_rf)
final_model_rf_grid.fit(X_train, y_train)

# Predict on the test set
y_pred_rf_g = final_model_rf_grid.predict(X_test)

accuracy_rf_g = accuracy_score(y_test, y_pred_rf_g)
print("Accuracy of Random Forest Classifier:", accuracy_rf_g)
print("Kfold cross validation Random Forest after Grid-----------------------------------------------")
kfold_cross_validation(random_forest_model,X_train,y_train)

cm_rf_g = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_rf_g, annot=True, cmap='summer', fmt='d')
plt.title('Confusion Matrix for Random Forest after grid search')
plt.show()

precision_rf_g= precision_score(y_test, y_pred_rf )
recall_rf_g = recall_score(y_test, y_pred_rf )
f_score_rf_g = f1_score(y_test, y_pred_rf )

# Display precision, recall, and F-score
print("Precision for Random Forest after grid search:", precision_rf_g.round(3))
print("Recall for Random Forest after grid search:", recall_rf_g.round(3))
print("F-score for Random Forest after grid search:", f_score_rf_g.round(3))
print("Specificity for Random Forest after grid search:", (cm_rf_g[0,0] / (cm_rf_g[0,0] + cm_rf_g[0,1])).round(3))
print("Confusion matrix :", cm_rf_g)
print("----------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf_g)
roc_auc_rf_g = roc_auc_score(y_test, y_pred_rf_g)


# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_rf_g)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest after grid search')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()





#---------------------------------Random Forest with Bagging----------------------------------------------------
rf_bagging = RandomForestClassifier(random_state=5805, n_jobs=-1)

bagging = BaggingClassifier(rf_bagging, n_estimators=100, random_state=5805, n_jobs=-1)
bagging.fit(X_train, y_train)

y_pred_rf_bag= bagging.predict(X_test)

accuracy_bg = accuracy_score(y_test, y_pred_rf)
print("Accuracy of Random Forest Classifier:", accuracy_bg.round(3))
print("Kfold cross validation Random Forest after Bagging-----------------------------------------------")
kfold_cross_validation(bagging,X_train,y_train)

cm_rf_bag = confusion_matrix(y_test, y_pred_rf_bag)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_rf_bag, annot=True, cmap='summer', fmt='d')
plt.title('Confusion Matrix for Random Forest after bagging')
plt.show()

precision_rf_b= precision_score(y_test, y_pred_rf_bag )
recall_rf_b = recall_score(y_test, y_pred_rf_bag )
f_score_rf_b = f1_score(y_test, y_pred_rf_bag )

# Display precision, recall, and F-score
print("Precision for Random Forest after bagging:", precision_rf_b.round(3))
print("Recall for Random Forest after bagging:", recall_rf_b.round(3))
print("F-score for Random Forest after bagging:", f_score_rf_b.round(3))
print("Specificity for Random Forest after bagging:", (cm_rf_bag[0,0] / (cm_rf_bag[0,0] + cm_rf_bag[0,1])).round(3))
print("Confusion matrix :", cm_rf_bag)
print("----------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf_bag)
roc_auc_rf_b = roc_auc_score(y_test, y_pred_rf_bag)


# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_rf_b)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest after bagging')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


#--------------------------------Random Forest with Stacking-----------------------------------------------------
rf_stacking = RandomForestClassifier(random_state=5805, n_jobs=-1)
final_model = RandomForestClassifier(random_state=5805, n_jobs=-1)

stacking = StackingClassifier(estimators=[('rfc', rf_stacking)], final_estimator=final_model, cv=5)
stacking.fit(X_train, y_train)

y_pred_rf_stack = stacking.predict(X_test)

accuracy_st = accuracy_score(y_test, y_pred_rf_stack)
print("Accuracy of Random Forest Classifier with stacking:", accuracy_st.round(3))
print("Kfold cross validation Random Forest after stacking-----------------------------------------------")
kfold_cross_validation(stacking,X_train,y_train)

cm_rf_st = confusion_matrix(y_test, y_pred_rf_stack)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_rf_st, annot=True, cmap='summer', fmt='d')
plt.title('Confusion Matrix for Random Forest after stacking')
plt.show()

precision_rf_s= precision_score(y_test, y_pred_rf )
recall_rf_s = recall_score(y_test, y_pred_rf )
f_score_rf_s = f1_score(y_test, y_pred_rf )

# Display precision, recall, and F-score
print("Precision for Random Forest after stacking:", precision_rf_s.round(3))
print("Recall for Random Forest after stacking:", recall_rf_s.round(3))
print("F-score for Random Forest after stacking:", f_score_rf_s.round(3))
print("Specificity for Random Forest after stacking:", (cm_rf_st[0,0] / (cm_rf_st[0,0] + cm_rf_st[0,1])).round(3))
print("Confusion matrix :", cm_rf_st)
print("----------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf_stack)
roc_auc_rf_s = roc_auc_score(y_test, y_pred_rf_stack)


# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_rf_s)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest after stacking')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


#-------------------------------Random Forest with boosting--------------------------------------------------

rf_boosting = RandomForestClassifier(random_state=5805, n_jobs=-1)

boosting = AdaBoostClassifier(rf_boosting, n_estimators=100, random_state=5805)
boosting.fit(X_train, y_train)

y_pred_rf_boost = boosting.predict(X_test)

accuracy_bt = accuracy_score(y_test, y_pred_rf_boost)
print("Accuracy of Random Forest Classifier with boosting:", accuracy_bt.round(3))


print("Kfold cross validation Random Forest after Boosting-----------------------------------------------")
kfold_cross_validation(boosting,X_train,y_train)

cm_rf_boost = confusion_matrix(y_test, y_pred_rf_boost)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_rf_boost, annot=True, cmap='summer', fmt='d')
plt.title('Confusion Matrix for Random Forest after boosting')
plt.show()

precision_rf_bb= precision_score(y_test, y_pred_rf_boost )
recall_rf_bb= recall_score(y_test, y_pred_rf_boost )
f_score_rf_bb = f1_score(y_test, y_pred_rf_boost )

# Display precision, recall, and F-score
print("Precision for Random Forest after boosting:", precision_rf_bb.round(3))
print("Recall for Random Forest after boosting:", recall_rf_bb.round(3))
print("F-score for Random Forest after boosting:", f_score_rf_bb.round(3))
print("Specificity for Random Forest after boosting:", (cm_rf_boost[0,0] / (cm_rf_boost[0,0] + cm_rf_boost[0,1])).round(3))
print("Confusion matrix :", cm_rf_boost)
print("----------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf_boost)
roc_auc_rf_bb = roc_auc_score(y_test, y_pred_rf_boost)


# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_rf_bb)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest after boosting')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


#
# #----------------------------------------------Neural Network multilayer perceptron--------------------------------

X_mlp = df_final[ ['OCCUPATION_TYPE_High skill tech staff', 'OCCUPATION_TYPE_Medicine staff', 'OCCUPATION_TYPE_Security staff', 'NAME_HOUSING_TYPE_With parents', 'FLAG_EMAIL',
                  'NAME_HOUSING_TYPE_House / apartment', 'NAME_FAMILY_STATUS_Separated', 'NAME_EDUCATION_TYPE_Incomplete higher',
                  'NAME_INCOME_TYPE_State servant', 'OCCUPATION_TYPE_Drivers', 'CODE_GENDER_M',
                  'OCCUPATION_TYPE_Managers', 'NAME_FAMILY_STATUS_Single / not married',
                  'NAME_FAMILY_STATUS_Married', 'NAME_INCOME_TYPE_Working', 'FLAG_OWN_CAR_Y',
                  'DAYS_EMPLOYED', 'OCCUPATION_TYPE_Core staff', 'FLAG_OWN_REALTY_Y', 'CNT_FAM_MEMBERS',
                  'FLAG_WORK_PHONE', 'FLAG_PHONE', 'OCCUPATION_TYPE_Sales staff',
                  'NAME_EDUCATION_TYPE_Secondary / secondary special', 'NAME_EDUCATION_TYPE_Higher education',
                  'OCCUPATION_TYPE_Laborers', 'AMT_INCOME_TOTAL']
]
y_mlp = df_final['response']

X_train, X_test, y_train, y_test = train_test_split(X_mlp, y_mlp, test_size=0.2, random_state=5805)

# Create a multi-layer perceptron neural network
neural_network_model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=100)

# Train the neural network
neural_network_model.fit(X_train, y_train)

# Predict the response for test dataset
y_pred_nn = neural_network_model.predict(X_test)

# Calculate the accuracy of the model
accuracy_nn = accuracy_score(y_test, y_pred_nn)
print("Kfold cross validation MLP--------------------------------")
kfold_cross_validation(neural_network_model,X_train,y_train)
print("Accuracy of Multi-Layer Perceptron Neural Network before grid:", accuracy_nn.round(3))

# Confusion matrix
cm_mlp = confusion_matrix(y_test, y_pred_nn)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_mlp, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix for MLP before grid')
plt.show()

precision_nn = precision_score(y_test, y_pred_nn )
recall_nn = recall_score(y_test, y_pred_nn )
f_score_nn = f1_score(y_test, y_pred_nn )

# Display precision, recall, and F-score
print("Precision for MLP before grid search:", precision_nn.round(3))
print("Recall for MLP before grid search:", recall_nn.round(3))
print("F-score for MLP before grid search:", f_score_nn.round(3))
print("Specificity for MLP before grid search:", (cm_mlp[0,0] / (cm_mlp[0,0] + cm_mlp[0,1])).round(3))
print("Confusion matrix :", cm_mlp)
print("----------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_nn)
roc_auc_nn = auc(fpr, tpr)

# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_nn)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve for MLP before grid search')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#Using tuning values to tune MLP---------------------------------
params_mlp = [{'hidden_layer_sizes':[(20,20)],'activation':['relu','tanh','logistic'],'learning_rate':['constant','adaptive']}]

grid_search_mlp = GridSearchCV(estimator=neural_network_model, param_grid=params_mlp, cv=5, scoring='accuracy')

# Fit the model
grid_search_mlp.fit(X_train, y_train)

# Get the best parameters
best_params_mlp = grid_search_mlp.best_params_

# Use the best parameters to train the final model
final_model_mlp = MLPClassifier(**best_params_mlp)
final_model_mlp.fit(X_train, y_train)

# Predict on the test set
y_pred_mlp_g = final_model_mlp.predict(X_test)

# Evaluate the accuracy
accuracy_mlp = accuracy_score(y_test, y_pred_mlp_g)
print("Accuracy of MLP classifier after grid:", accuracy_mlp.round(3))
print("Kfold cross validation MLP--------------------------------")
kfold_cross_validation(final_model_mlp,X_train,y_train)

# Confusion matrix
cm_mlp_g = confusion_matrix(y_test, y_pred_mlp_g)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_mlp_g, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix for MLP after grid')
plt.show()

precision_mlp = precision_score(y_test, y_pred_mlp_g )
recall_mlp = recall_score(y_test, y_pred_mlp_g )
f_score_mlp = f1_score(y_test, y_pred_mlp_g )

# Display precision, recall, and F-score
print("Precision for MLP after grid search:", precision_mlp.round(3))
print("Recall for MLP after grid search:", recall_mlp.round(3))
print("F-score for MLP after grid search:", f_score_mlp.round(3))
print("Specificity for MLP after grid search:", (cm_mlp_g[0,0] / (cm_mlp_g[0,0] + cm_mlp_g[0,1])).round(3))
print("Confusion matrix :", cm_mlp_g)
print("----------------------------------------------------------------\n")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_mlp_g)
roc_auc_nn = auc(fpr, tpr)
auc_mlp = roc_auc_score(y_test, y_pred_mlp_g)


# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_nn)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve for MLP after grid search')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


#------------------------------Table showing the performance of all the classifiers------------------------------------

table = PrettyTable()

table.field_names = ["Classifier", "Accuracy", "Precision", "Recall", "F-Score", "Specificity"]

table.add_row(["Decision Tree", accuracy_dt.round(3), precision_dt.round(3), recall_dt.round(3), f_score_dt.round(3), (cm_dt[0,0] / (cm_dt[0,0] + cm_dt[0,1])).round(3)])

table.add_row(["Decision Tree pre-pruned", accuracy_dt_prepruned.round(3), precision_dt_pp.round(3), recall_dt_pp.round(3), f_score_dt_pp.round(3), (cm_pp[0,0] / (cm_pp[0,0] + cm_pp[0,1])).round(3)])

table.add_row(["Decision Tree post-pruned", accuracy_dt_ccp.round(3), precision_dt_ccp.round(3), recall_dt_ccp.round(3), f_score_dt_ccp.round(3), (cm_ppp[0,0] / (cm_ppp[0,0] + cm_ppp[0,1])).round(3)])

table.add_row(["Logistic Regression", accuracy_lr.round(3), precision_lr.round(3), recall_lr.round(3), f_score_lr.round(3), (cm_lr[0,0] / (cm_lr[0,0] + cm_lr[0,1])).round(3)])

table.add_row(["Logistic Regression after Grid Search", accuracy_lr_g.round(3), precision_lr_g.round(3), recall_lr_g.round(3), f_score_lr_g.round(3), (cm_lr_g[0,0] / (cm_lr_g[0,0] + cm_lr_g[0,1])).round(3)])

table.add_row(["KNN", accuracy_knn.round(3), precision_knn.round(3), recall_knn.round(3), f_score_knn.round(3), (cm_knn[0,0] / (cm_knn[0,0] + cm_knn[0,1])).round(3)])

table.add_row(["KNN after Grid Search", accuracy_knn_g.round(3), precision_knn_g.round(3), recall_knn_g.round(3), f_score_knn_g.round(3), (cm_knn_g[0,0] / (cm_knn_g[0,0] + cm_knn_g[0,1])).round(3)])

table.add_row(["SVM", accuracy_svm.round(3), precision_svm.round(3), recall_svm.round(3), f_score_svm.round(3), (cm_svm[0,0] / (cm_svm[0,0] + cm_svm[0,1])).round(3)])

table.add_row(["SVM after Grid Search", accuracy_svm_grid.round(3), precision_svm_grid.round(3), recall_svm_grid.round(3), f_score_svm_grid.round(3), (cm_svm_grid[0,0] / (cm_svm_grid[0,0] + cm_svm_grid[0,1])).round(3)])

table.add_row(["Naive Bayes", accuracy_nb.round(3), precision_nb.round(3), recall_nb.round(3), f_score_nb.round(3), (cm_nb[0,0] / (cm_nb[0,0] + cm_nb[0,1])).round(3)])

table.add_row(["Naive Bayes after Grid Search", accuracy_nb_g.round(3), precision_nb_g.round(3), recall_nb_g.round(3), f_score_nb_g.round(3), (cm_nb_g[0,0] / (cm_nb_g[0,0] + cm_nb_g[0,1])).round(3)])

table.add_row(["Random Forest", accuracy_rf.round(3), precision_rf.round(3), recall_rf.round(3), f_score_rf.round(3), (cm_rf[0,0] / (cm_rf[0,0] + cm_rf[0,1])).round(3)])

table.add_row(["Random Forest after Grid Search", accuracy_rf_g.round(3), precision_rf_g.round(3), recall_rf_g.round(3), f_score_rf_g.round(3), (cm_rf_g[0,0] / (cm_rf_g[0,0] + cm_rf_g[0,1])).round(3)])

table.add_row(["Random Forest after Bagging", accuracy_bg.round(3), precision_rf_b.round(3), recall_rf_b.round(3), f_score_rf_b.round(3), (cm_rf_bag[0,0] / (cm_rf_bag[0,0] + cm_rf_bag[0,1])).round(3)])

table.add_row(["Random Forest after Boosting", accuracy_bt.round(3), precision_rf_bb.round(3), recall_rf_bb.round(3), f_score_rf_bb.round(3), (cm_rf_boost[0,0] / (cm_rf_boost[0,0] + cm_rf_boost[0,1])).round(3)])

table.add_row(["Random Forest after Stacking", accuracy_st.round(3), precision_rf_s.round(3), recall_rf_s.round(3), f_score_rf_s.round(3), (cm_rf_st[0,0] / (cm_rf_st[0,0] + cm_rf_st[0,1])).round(3)])

table.add_row(["Multilayer Perceptron", accuracy_nn.round(3), precision_nn.round(3), recall_nn.round(3), f_score_nn.round(3), (cm_mlp[0,0] / (cm_mlp[0,0] + cm_mlp[0,1])).round(3)])

table.add_row(["Multilayer Perceptron with Grid Search", accuracy_mlp.round(3), precision_mlp.round(3), recall_mlp.round(3), f_score_mlp.round(3), (cm_mlp_g[0,0] / (cm_mlp_g[0,0] + cm_mlp_g[0,1])).round(3)])


print(table)














