import pickle
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

# bring data
credit_dummy = pd.read_csv(r"C:\Users\Dell\Documents\Python Credit Risk\Data\credit_downsampled_dummy.csv", low_memory=False,
                           index_col=0)

# Create test and train datasets
credit_dummy = credit_dummy.astype(int)
y = credit_dummy.MIS_Status
X = credit_dummy.drop(['MIS_Status'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# ------------------------------------------------------- RANDOM FOREST
# OOB Random Forest Model
# Check if exists already if not train
try:
    rf_OOB = pickle.load(open('rf_OOB.sav', 'rb'))
except FileNotFoundError:
    print("File doesen't exist, will train the model")
    # Build model
    rf_OOB = RandomForestClassifier()
    rf_OOB.fit(X_train, y_train)
    # Save the model to disk
    pickle.dump(rf_OOB, open('rf_OOB.sav', 'wb'))

#  predict and performance
rf_OOB_predict = rf_OOB.predict(X_test)
rfc_cv_score = cross_val_score(rf_OOB, X, y, cv=10, scoring="roc_auc")
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rf_OOB_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rf_OOB_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
print("Accuracy for model: %.2f" % (accuracy_score(y_test, rf_OOB_predict) * 100), ' % ')  # Accuracy for model: 64  %

# Feature importance
print("Features sorted by their score:")
feats = {}  # a dict to hold feature_name: feature_importance
for feature, importance in zip(X_test.columns, rf_OOB.feature_importances_):
    feats[feature] = importance  # add the name/value pair

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=45)
print(importances.sort_values(by='Gini-importance', ascending=False))

# Hyperparameter tuning

# number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

# number of features at every split
max_features = ['auto', 'sqrt']

# max depth
max_depth = [int(x) for x in np.linspace(100, 500, num=11)]
max_depth.append(None)

# create random grid
random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth
}

# Random search of parameters
rfc_random = RandomizedSearchCV(estimator=rf_OOB, param_distributions=random_grid, n_iter=5, cv=3, verbose=2,
                                random_state=42, n_jobs=3)

# Fit the model
rfc_random.fit(X_train, y_train)

# print results
print(rfc_random.best_params_)  # {'n_estimators': 1400, 'max_features': 'auto', 'max_depth': 260}

# Tuned model Random Forest
# Check if exists already if not train
try:
    rf_tuned = pickle.load(open('rf_tuned.sav', 'rb'))
except FileNotFoundError:
    print("File doesen't exist, will train the model")
    # Build model with chosen hyperparamenters
    rf_tuned = RandomForestClassifier(n_estimators=1400, max_depth=260, max_features='auto')
    rf_tuned.fit(X_train, y_train)
    # Save the model to disk
    pickle.dump(rf_tuned, open('rf_tuned.sav', 'wb'))

#  predict and performance
rf_tuned_predict = rf_tuned.predict(X_test)
rf_tuned_cv_score = cross_val_score(rf_tuned, X, y, cv=10, scoring='roc_auc', n_jobs=3)
prob_rf = rf_tuned.predict_proba(X_test)
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rf_tuned_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rf_tuned_predict))
print('\n')
print("=== All AUC Scores ===")
print(rf_tuned_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rf_tuned_cv_score.mean())
print("Accuracy for model: %.2f" % (accuracy_score(y_test, rf_tuned_predict) * 100))  # Accuracy for model: 60 %

# ------------------------------------------------------- SVM
# Train a linear SVM model using linear kernel
svm_OOB_linear = sklearn.svm.LinearSVC(max_iter=10000)
svm_OOB_linear.fit(X_train, y_train)

# Make prediction
svm_OOB_linear_pred = svm_OOB_linear.predict(X_test)

# Evaluate our model
print("Evaluation linear kernel")
print(classification_report(y_test, svm_OOB_linear_pred))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, svm_OOB_linear_pred))  # 63 % accuracy
pickle.dump(svm_OOB_linear, open('svm_OOB_linear.sav', 'wb'))

# ------------------------------------------------------- KNN
classifier = KNeighborsClassifier(n_neighbors=5, n_jobs = 3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy for model: %", (accuracy_score(y_test, y_pred) * 100))
# 60 % accuracy

# Try to find best number neighbours
error = []

# Calculating error for K values between 1 and 10
for i in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

# Plot the error for the number neighbours
plt.figure(figsize=(12, 6))
plt.plot(range(1, 100), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
# Best knn 97

# KNN 97 neighbours model
knn_97 = KNeighborsClassifier(n_neighbors=97, n_jobs=3)
knn_97.fit(X_train, y_train)
y_pred_knn_97 = knn_97.predict(X_test)
prob_knn = knn_97.predict_proba(X_test)
print(confusion_matrix(y_test, y_pred_knn_97))
print(classification_report(y_test, y_pred_knn_97))
print("Accuracy for model: %", (accuracy_score(y_test, y_pred_knn_97) * 100))
# 64.42 % accuracy
pickle.dump(knn_97, open('knn_97.sav', 'wb'))

rf_tuned = pickle.load(open('rf_tuned.sav', 'rb'))
knn_97 = pickle.load(open('knn_97.sav', 'rb'))
svm_OOB_linear = pickle.load(open('svm_OOB_linear.sav', 'rb'))

# At this point we tried 3 models Random Forest, SVM and KNN with K = 97,
# they all have an accuracy close to 60 % minus the svm
# We will try a model ensemble in order to better the predictions with knn and rf,svm
# Averaging
final_pred_average = (prob_knn + prob_rf) / 2
final_pred_average_df = pd.DataFrame()
final_pred_average_df["pred_0"], final_pred_average_df["pred_1"] = final_pred_average.T
final_pred_average_df['Prediction'] = final_pred_average_df.idxmax(axis=1)

dct = {"pred_0": 0,
       'pred_1': 1}

final_pred_average_df = final_pred_average_df.assign(Prediction=final_pred_average_df.Prediction.map(dct))
final_pred_average_df = final_pred_average_df.drop(columns=["pred_0", "pred_1"])

# Compare the results of the average with the test values
y_test_df = pd.DataFrame(y_test)
y_test_df.index = np.arange(1, len(y_test_df) + 1)
final_pred_average_df.index = np.arange(1, len(final_pred_average_df) + 1)
final_pred_average_df = pd.concat([final_pred_average_df, y_test_df], axis=1)
final_pred_average_df = final_pred_average_df.astype(int)
final_pred_average_df['result'] = np.where(final_pred_average_df['Prediction'] == final_pred_average_df['MIS_Status'],
                                           "correct", "incorrect")
Counter(final_pred_average_df["result"])