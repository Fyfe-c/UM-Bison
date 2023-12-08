import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Data loading and preprocessing
df = pd.read_csv('./updated_weather_data_with_states.csv')

# Dropping non-numeric columns and columns with a large number of missing values
columns_to_drop = ['name', 'solarradiation', 'solarenergy', 'uvindex', 'severerisk', 'snow']
df = df.drop(columns=columns_to_drop, axis=1)

# Converting the date column and sorting by date
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(by='datetime')

# Filling missing values
# df.fillna(df.mean(), inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Splitting the data using a new split date
split_date = pd.Timestamp('2006-01-01')
train = df[df['datetime'] < split_date]
test = df[df['datetime'] >= split_date]

# Splitting features and labels
X_train = train.select_dtypes(include=[np.number]).drop('result', axis=1)
y_train = train['result']
X_test = test.select_dtypes(include=[np.number]).drop('result', axis=1)
y_test = test['result']

# Applying standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Addressing data imbalance using SMOTE
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Building and training the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_smote, y_train_smote)

# Model evaluation
y_pred = model.predict(X_test_scaled)
print("Test accuracy:", model.score(X_test_scaled, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

'''
# Plotting the confusion matrix heatmap using Seaborn
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
'''

'''
# Calculating ROC curve
y_score = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
'''