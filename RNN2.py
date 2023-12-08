import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix

# Data loading and preprocessing
df = pd.read_csv('./updated_weather_data_with_states.csv')
# df = pd.read_csv('./Data/Data/Test.csv')

# Dropping non-numeric columns and columns with a large number of missing values
# columns_to_drop = ['name', 'solarradiation', 'solarenergy', 'uvindex', 'severerisk', 'snow','sunset','sunrise','moonphase','conditions','description','icon','stations','visibility','cloudcover','sealevelpressure','winddir','windgust','snowdepth','preciptype','precipcover','precipcover','precip']
columns_to_drop = ['name', 'solarradiation', 'solarenergy', 'uvindex', 'severerisk', 'snow']
df = df.drop(columns=columns_to_drop, axis=1)

# Converting the date column and sorting by date
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(by='datetime')

# Filling missing values
df.fillna(df.mean(), inplace=True)

# Splitting the data using a new split date
split_date = pd.Timestamp('2006-01-01')  # Update this date according to your dataset
train = df[df['datetime'] < split_date]
test = df[df['datetime'] >= split_date]

# Checking the size of the datasets
print("Training set size:", train.shape)
print("Test set size:", test.shape)

# Splitting features and labels
X_train = train.select_dtypes(include=[np.number]).drop('result', axis=1)
y_train = train['result']
X_test = test.select_dtypes(include=[np.number]).drop('result', axis=1)
y_test = test['result']

# Applying standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshaping for RNN format
X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_rnn = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Addressing data imbalance using SMOTE
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train_rnn.reshape(X_train_rnn.shape[0], -1), y_train)
X_train_smote = X_train_smote.reshape((X_train_smote.shape[0], 1, X_train_smote.shape[1]))

# Building LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train_smote.shape[1], X_train_smote.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Training the model
model.fit(X_train_smote, y_train_smote, epochs=10, batch_size=32, validation_data=(X_test_rnn, y_test))

# Model evaluation
score = model.evaluate(X_test_rnn, y_test, verbose=0)
print('Test accuracy:', score[1])

# Display confusion matrix and classification report
y_pred = (model.predict(X_test_rnn) > 0.5).astype("int32")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
