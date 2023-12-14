import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score

# Data loading and preprocessing
df = pd.read_csv('./updated_weather_data_with_states.csv')

# Dropping non-numeric columns and columns with a large number of missing values
columns_to_drop = ['name', 'solarradiation', 'solarenergy', 'uvindex', 'severerisk', 'snow']
df = df.drop(columns=columns_to_drop, axis=1)

# Converting the date column and sorting by date
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(by='datetime')

# Filling missing values
df.fillna(df.mean(), inplace=True)

# Splitting features and labels
X = df.select_dtypes(include=[np.number]).drop('result', axis=1)
y = df['result']

# Applying standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshaping for RNN format
X_rnn = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Addressing data imbalance using SMOTE
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X_rnn.reshape(X_rnn.shape[0], -1), y)
X_smote = X_smote.reshape((X_smote.shape[0], 1, X_smote.shape[1]))

# Define the LSTM model creation function
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# KFold cross-validation
kfold = KFold(n_splits=10, shuffle=True)
scores = []

for train, test in kfold.split(X_smote, y_smote):
    # Create the model
    model = create_model((X_smote.shape[1], X_smote.shape[2]))

    # Train the model
    model.fit(X_smote[train], y_smote[train], epochs=10, batch_size=32, verbose=0)

    # Evaluate the model
    predictions = model.predict(X_smote[test])
    y_pred = (predictions > 0.5).astype("int32")
    score = accuracy_score(y_smote[test], y_pred)
    scores.append(score)

# Print cross-validation results
print("Cross-Validation Accuracy: %.2f%% (%.2f%%)" % (np.mean(scores)*100, np.std(scores)*100))
