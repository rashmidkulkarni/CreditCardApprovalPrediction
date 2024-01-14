cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, cmap='summer', fmt='d')
plt.title('Confusion Matrix for Random Forest before grid search')
plt.show()

precision_rf= precision_score(y_test, y_pred_rf )
recall_rf = recall_score(y_test, y_pred_rf )
f_score_rf = f1_score(y_test, y_pred_rf )

# Display precision, recall, and F-score
print("Precision for Random Forest before grid search:", precision_rf.round(3))
print("Recall for Random Forest before grid search:", recall_rf.round(3))
print("F-score for Random Forest before grid search:", f_score_rf.round(3))

fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)


# Display ROC curve and AUC
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Naive Bayes after grid search')
plt.legend(loc="lower right")
plt.show()
