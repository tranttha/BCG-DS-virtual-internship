# # Make a copy of our data
train_df = df.copy()

# Separate target variable from independent variables
y = df['churn']
X = df.drop(columns=['id', 'churn'])
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

y_scores_thres = model_rfc.predict_proba(X_test)[:, 1]
precision_thres, recall_thres, thresholds = precision_recall_curve(y_test, y_scores_thres)
# Calculate F1 score for each threshold
# f1_scores_thres = 2 * (precision_thres * recall_thres) / (precision_thres + recall_thres)
epsilon = 1e-10  # Small constant to avoid division by zero

# Calculate F2 score for each threshold
precision_thres_safe = precision_thres + epsilon
recall_thres_safe = recall_thres + epsilon

f1_5_scores_sm = ((1 + 1.25**2) / ((1.25**2 / recall_thres_safe) + (1 / precision_thres_safe)))

# Get the index of the highest F1 score
best_idx = np.argmax(f1_5_scores_sm)
best_threshold = thresholds[best_idx]

print(f"Best Threshold: {best_threshold}")
print(f"Best F1 Score: {f1_5_scores_sm[best_idx]}")
y_pred_thres = (y_scores_thres >= best_threshold).astype(int)
precision_best = precision_score(y_test, y_pred_thres)
recall_best = recall_score(y_test, y_pred_thres)
f1_best = f1_score(y_test, y_pred_thres)
print(f"Precision: {precision_best:.2f}")
print(f"Recall: {recall_best:.2f}")
print(f"F1 Score: {f1_best:.2f}")

# Detailed classification report
print("\nClassification Report:\n")
# Print classification report
print(f"{classification_report(y_test, y_pred_thres)}")

# Confusion matrix
cm_thres = confusion_matrix(y_test, y_pred_thres)
print(f"Confusion Matrix:\n{cm_thres}")

# Plot the precision-recall curve with the optimal threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision_thres[:-1], label="Precision")
plt.plot(thresholds, recall_thres[:-1], label="Recall")
plt.axvline(x=best_threshold, color='r', linestyle='--', label="Best Threshold")
plt.xlabel("Threshold")
plt.ylabel("Precision/Recall")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

Precision: 0.09
Recall: 0.18
F1 Score: 0.12

Classification Report:

              precision    recall  f1-score   support

           0       0.90      0.81      0.85      3297
           1       0.09      0.18      0.12       355

    accuracy                           0.75      3652
   macro avg       0.50      0.50      0.49      3652
weighted avg       0.82      0.75      0.78      3652

Confusion Matrix:
[[2677  620]
 [ 291   64]]

# plt.figure(figsize=(6,4))
# sns.heatmap(cm_thres, annot=True, fmt='d', cmap='Blues', cbar=False,
#             xticklabels=['Predicted 0', 'Predicted 1'],
#             yticklabels=['Actual 0', 'Actual 1'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

4.2 undersampling
y = df['churn']
X = df.drop(columns=['id', 'churn'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train_majority = X_train[y_train == 0]
y_train_majority = y_train[y_train == 0]
X_train_minority = X_train[y_train == 1]
y_train_minority = y_train[y_train == 1]

n_estimators = 10  # Number of RFM in the ensemble
undersample_size = len(y_train_minority)  # Size of each undersampled subset
models = []

# Train a separate model on each undersampled subset
for i in range(n_estimators):
    # Randomly undersample the majority class
    X_train_undersampled = X_train_majority.sample(n=undersample_size, random_state=i)
    y_train_undersampled = y_train_majority.sample(n=undersample_size, random_state=i)

    # Combine the minority and undersampled majority samples
    X_train_combined = pd.concat([X_train_undersampled, X_train_minority], axis=0)
    y_train_combined = pd.concat([y_train_undersampled, y_train_minority], axis=0)

    # Create and train the model
    model_us = RandomForestClassifier(n_estimators=1000, random_state=i)
    model_us.fit(X_train_combined, y_train_combined)

    # Append the model_us to the list
    models.append(model_us)

# Function to make predictions using the ensemble of models
def ensemble_predict(models, X):
    # Get predictions from all models
    predictions = np.array([model.predict(X) for model in models])
    # Majority vote
    majority_vote = np.mean(predictions, axis=0) >= 0.5
    return majority_vote.astype(int)

y_pred_us = ensemble_predict(models, X_test)

# Calculate precision, recall_us, and f1 score
precision_us = precision_score(y_test, y_pred_us)
recall_us = recall_score(y_test, y_pred_us)
f1_us = f1_score(y_test, y_pred_us)

print(f"Precision: {precision_us:.2f}")
print(f"Recall: {recall_us:.2f}")
print(f"F1 Score: {f1_us:.2f}")

# Detailed classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_us))

Precision: 0.15
Recall: 0.62
F1 Score: 0.24

Classification Report:

              precision    recall  f1-score   support

           0       0.94      0.62      0.74      3286
           1       0.15      0.62      0.24       366

    accuracy                           0.62      3652
   macro avg       0.54      0.62      0.49      3652
weighted avg       0.86      0.62      0.69      3652

cm_us = confusion_matrix(y_test, y_pred_us)
plt.figure(figsize=(6,4))
sns.heatmap(cm_us, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

4.2b threshold tunning of undersampling 
# Function to get prediction probabilities from the ensemble
def ensemble_predict_proba(models, X):
    # Get prediction probabilities from all models
    probabilities = np.array([model.predict_proba(X)[:, 1] for model in models])
    # Average the probabilities across all models
    mean_probabilities = np.mean(probabilities, axis=0)
    return mean_probabilities

# Get the prediction probabilities for the test set
y_scores_us = ensemble_predict_proba(models, X_test)
precision_us_thres, recall_us_thres, thresholds_us = precision_recall_curve(y_test, y_scores_us)

# f1_scores_us_thres = 2 * (precision_us_thres * recall_us_thres) / (precision_us_thres + recall_us_thres + 1e-10)  # Adding a small value to avoid division by zero
# Get the index of the highest F1 score
epsilon = 1e-10  # Small constant to avoid division by zero

# Calculate F2 score for each threshold
precision_us_thres_safe = precision_us_thres + epsilon
recall_us_thres_safe = recall_us_thres + epsilon

f1_5_scores_us = ((1 + 1.15**2) / ((1.15**2 / recall_us_thres_safe) + (1 / precision_us_thres_safe)))

best_idx_us = np.argmax(f1_5_scores_us)
best_threshold_us = thresholds_us[best_idx_us]

print(f"Best Threshold: {best_threshold_us:.4f}")
print(f"Best F1 Score: {f1_5_scores_us[best_idx_us]:.4f}")

Best Threshold: 0.7455
Best F1 Score: 0.8328

y_pred_us_best = (y_scores_us >= best_threshold_us).astype(int)

# Calculate metrics for the predictions
precision_us_best = precision_score(y_test, y_pred_us_best)
recall_us_best = recall_score(y_test, y_pred_us_best)
f1_us_best = f1_score(y_test, y_pred_us_best)

print(f"Precision (Best Threshold): {precision_us_best:.2f}")
# print(f"Recall (Best Threshold): {recall_us_best:.2f}")
print(f"F1 Score (Best Threshold): {f1_us_best:.2f}")

# Detailed classification report
print("\nClassification Report (Best Threshold):\n")
print(classification_report(y_test, y_pred_us_best))

cm_us_thres = confusion_matrix(y_test, y_pred_us_best)
print(f"Confusion Matrix:\n{cm_us_thres}")

# Plot the precision-recall curve with the optimal threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds_us, precision_us_thres[:-1], label="Precision")
plt.plot(thresholds_us, recall_us_thres[:-1], label="Recall")
plt.axvline(x=best_threshold_us, color='r', linestyle='--', label="Best Threshold")
plt.xlabel("Threshold")
plt.ylabel("Precision/Recall")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

Precision (Best Threshold): 0.91
Recall (Best Threshold): 0.78
F1 Score (Best Threshold): 0.84

Classification Report (Best Threshold):

              precision    recall  f1-score   support

           0       0.98      0.99      0.98      3297
           1       0.91      0.78      0.84       355

    accuracy                           0.97      3652
   macro avg       0.95      0.89      0.91      3652
weighted avg       0.97      0.97      0.97      3652

Confusion Matrix:
[[3271   26]
 [  78  277]]
** !!!! leaked through other method**

4.3. Oversampling with SMOTE 
from imblearn.over_sampling import SMOTE
y = df['churn']
X = df.drop(columns=['id', 'churn'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
class_counts = y_test.value_counts()
class_counts
# Applying SMOTE to the training data
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
# Training the Random Forest classifier on the resampled data
clf = RandomForestClassifier(n_estimators=1000, random_state=42)
clf.fit(X_train_resampled, y_train_resampled)
# Making predictions on the test set
y_pred_sm = clf.predict(X_test)

# Calculating precision, recall, and F1 score
precision_sm = precision_score(y_test, y_pred_sm)
recall_sm = recall_score(y_test, y_pred_sm)
f1_sm = f1_score(y_test, y_pred_sm)

print(f"Precision: {precision_sm:.2f}")
print(f"Recall: {recall_sm:.2f}")
print(f"F1 Score: {f1_sm:.2f}")

# Detailed classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_sm))
Precision: 0.59
Recall: 0.13
F1 Score: 0.22

Classification Report:

              precision    recall  f1-score   support

           0       0.91      0.99      0.95      3297
           1       0.59      0.13      0.22       355

    accuracy                           0.91      3652
   macro avg       0.75      0.56      0.58      3652
weighted avg       0.88      0.91      0.88      3652
cm_sm = confusion_matrix(y_test, y_pred_sm)
# Create a DataFrame from the confusion matrix for better visualization
cm_df = pd.DataFrame(cm_sm, index=['Negative', 'Positive'], columns=['Predicted Negative', 'Predicted Positive'])

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True, 
            annot_kws={"size": 16}, 
            linewidths=.5, 
            linecolor='black')

plt.title('Confusion Matrix Heatmap')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

4.3.b threshold tunning of oversampling 
y_scores_sm = clf.predict_proba(X_test)[:, 1]
precision_sm_thres, recall_sm_thres, thresholds_sm = precision_recall_curve(y_test, y_scores_sm)
# Calculate F1 score for each threshold
# f1_scores_sm = 2 * (precision_sm_thres * recall_sm_thres) / (precision_sm_thres + recall_sm_thres)
epsilon = 1e-10  # Small constant to avoid division by zero

# Calculate F2 score for each threshold
precision_sm_thres_safe = precision_sm_thres + epsilon
recall_sm_thres_safe = recall_sm_thres + epsilon

f1_5_scores_sm = ((1 + 1.15**2) / ((1.15**2 / recall_sm_thres_safe) + (1 / precision_sm_thres_safe)))

# Get the index of the highest F1 score
best_idx_sm = np.argmax(f1_5_scores_sm)
best_threshold_sm = thresholds_sm[best_idx_sm]

print(f"Best Threshold: {best_threshold_sm}")
print(f"Best F1 Score: {f1_5_scores_sm[best_idx_sm]}")
y_pred_sm_best = (y_scores_sm >= best_threshold_sm).astype(int)

# Calculate metrics for the predictions
precision_sm_best = precision_score(y_test, y_pred_sm_best)
recall_sm_best = recall_score(y_test, y_pred_sm_best)
f1_5_scores_sm = f1_score(y_test, y_pred_sm_best)

print(f"Precision (Best Threshold): {precision_sm_best:.2f}")
print(f"Recall (Best Threshold): {recall_sm_best:.2f}")
print(f"F1 Score (Best Threshold): {f1_5_scores_sm:.2f}")

print(classification_report(y_test, y_pred_sm_best))

# Confusion matrix
cm_sm_thres = confusion_matrix(y_test, y_pred_sm_best)
print(f"Confusion Matrix:\n{cm_sm_thres}")

# Plot the precision-recall curve with the optimal threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds_sm, precision_sm_thres[:-1], label="Precision")
plt.plot(thresholds_sm, recall_sm_thres[:-1], label="Recall")
plt.axvline(x=best_threshold_sm, color='r', linestyle='--', label="Best Threshold")
plt.xlabel("Threshold")
plt.ylabel("Precision/Recall")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

Precision (Best Threshold): 0.25
Recall (Best Threshold): 0.37
F1 Score (Best Threshold): 0.30
              precision    recall  f1-score   support

           0       0.93      0.88      0.90      3297
           1       0.25      0.37      0.30       355

    accuracy                           0.83      3652
   macro avg       0.59      0.63      0.60      3652
weighted avg       0.86      0.83      0.84      3652

Confusion Matrix:
[[2894  403]
 [ 222  133]]

plt.figure(figsize=(6,4))
sns.heatmap(cm_sm_thres, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
result:

- traditional RFC (nesitmator = 1000)

Metrics:
       
Precision: 0.86
Recall: 0.05
F1 Score: 0.09

Classification Report:

       precision    recall  f1-score   support

    0       0.90      1.00      0.95      3286
    1       0.86      0.05      0.09       366

accuracy                           0.90      3652
macro avg       0.88      0.52      0.52      3652
weighted avg       0.90      0.90      0.86      3652


F1 Score: 0.09
F1.5 Score: 0.07
F2 Score: 0.06
F3 Score: 0.05
b. RFC with threshold tuning 

Traditional RFC with Threshold Tuning - Precision: 0.26, Recall: 0.33, F1 Score: 0.29

Classification Report (Best Threshold):

              precision    recall  f1-score   support

           0       0.92      0.90      0.91      3286
           1       0.26      0.33      0.29       366

    accuracy                           0.84      3652
   macro avg       0.59      0.61      0.60      3652
weighted avg       0.86      0.84      0.85      3652
F1 Score: 0.29
F1.5 Score: 0.31
F2 Score: 0.31
F3 Score: 0.32

c. SMOTE with Threshold Tuning
SMOTE with Threshold Tuning - Precision: 0.23, Recall: 0.31, F1 Score: 0.27

Classification Report (Best Threshold):

              precision    recall  f1-score   support

           0       0.92      0.89      0.90      3286
           1       0.23      0.31      0.27       366

    accuracy                           0.83      3652
   macro avg       0.58      0.60      0.59      3652
weighted avg       0.85      0.83      0.84      3652
F1 Score: 0.27
F1.5 Score: 0.28
F2 Score: 0.29
F3 Score: 0.30



d. Random Undersampling Ensemble with Threshold Tuning

/Users/hoangtran/miniconda3/envs/localenv/lib/python3.10/site-packages/sklearn/base.py:486: UserWarning: X has feature names, but RandomForestClassifier was fitted without feature names
  warnings.warn(
/Users/hoangtran/miniconda3/envs/localenv/lib/python3.10/site-packages/sklearn/base.py:486: UserWarning: X has feature names, but RandomForestClassifier was fitted without feature names
  warnings.warn(
/Users/hoangtran/miniconda3/envs/localenv/lib/python3.10/site-packages/sklearn/base.py:486: UserWarning: X has feature names, but RandomForestClassifier was fitted without feature names
  warnings.warn(
/Users/hoangtran/miniconda3/envs/localenv/lib/python3.10/site-packages/sklearn/base.py:486: UserWarning: X has feature names, but RandomForestClassifier was fitted without feature names
  warnings.warn(
/Users/hoangtran/miniconda3/envs/localenv/lib/python3.10/site-packages/sklearn/base.py:486: UserWarning: X has feature names, but RandomForestClassifier was fitted without feature names
  warnings.warn(
/Users/hoangtran/miniconda3/envs/localenv/lib/python3.10/site-packages/sklearn/base.py:486: UserWarning: X has feature names, but RandomForestClassifier was fitted without feature names
  warnings.warn(
/Users/hoangtran/miniconda3/envs/localenv/lib/python3.10/site-packages/sklearn/base.py:486: UserWarning: X has feature names, but RandomForestClassifier was fitted without feature names
  warnings.warn(
/Users/hoangtran/miniconda3/envs/localenv/lib/python3.10/site-packages/sklearn/base.py:486: UserWarning: X has feature names, but RandomForestClassifier was fitted without feature names
  warnings.warn(
/Users/hoangtran/miniconda3/envs/localenv/lib/python3.10/site-packages/sklearn/base.py:486: UserWarning: X has feature names, but RandomForestClassifier was fitted without feature names
  warnings.warn(
/Users/hoangtran/miniconda3/envs/localenv/lib/python3.10/site-packages/sklearn/base.py:486: UserWarning: X has feature names, but RandomForestClassifier was fitted without feature names
  warnings.warn(
Best Threshold: 0.5468
Best F1 Score: 0.2871
Precision (Best Threshold): 0.18
Recall (Best Threshold): 0.51
F1 Score (Best Threshold): 0.27

Classification Report (Best Threshold):

              precision    recall  f1-score   support

           0       0.93      0.74      0.83      3286
           1       0.18      0.51      0.27       366

    accuracy                           0.72      3652
   macro avg       0.56      0.63      0.55      3652
weighted avg       0.86      0.72      0.77      3652
F1 Score: 0.27
F1.5 Score: 0.33
F2 Score: 0.38
F3 Score: 0.43
