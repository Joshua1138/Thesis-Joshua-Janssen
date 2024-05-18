# Logistic Regression

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load datasets
train_df = pd.read_csv('train.csv', sep=';', names=['text', 'label'], header=0)
val_df = pd.read_csv('val.csv', sep=';', names=['text', 'label'], header=0)
test_df = pd.read_csv('test.csv', sep=';', names=['text', 'label'], header=0)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['text'])
X_val = vectorizer.transform(val_df['text'])
X_test = vectorizer.transform(test_df['text'])

y_train = train_df['label']
y_val = val_df['label']
y_test = test_df['label']
# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Validate the model
predictions_val = model.predict(X_val)
print('Validation Results:')
print(classification_report(y_val, predictions_val))

# Test the model
predictions_test = model.predict(X_test)
print('Test Results:')
print(classification_report(y_test, predictions_test))

# Save the confusion matrix
cm_test = confusion_matrix(y_test, predictions_test)
pd.DataFrame(cm_test).to_csv('confusion_matrix_test.csv', index=False)
print('Testing confusion matrix saved as confusion_matrix_test.csv')
