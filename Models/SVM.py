# SVM

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Function to load datasets
def load_dataset(filename):
    return pd.read_csv(filename, sep=';', header=0, names=['text', 'label'])

# Load datasets
train_data = load_dataset('train.csv')
val_data = load_dataset('val.csv')
test_data = load_dataset('test.csv')

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training text data
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['text'])

# Only transform the validation and test text data
X_val_tfidf = tfidf_vectorizer.transform(val_data['text'])
X_test_tfidf = tfidf_vectorizer.transform(test_data['text'])

# Extract labels
y_train = train_data['label']
y_val = val_data['label']
y_test = test_data['label']

# Initialize the SVM classifier with balanced class weights
svm_classifier = SVC(kernel='rbf', class_weight='balanced')

# Train the classifier
svm_classifier.fit(X_train_tfidf, y_train)

# Predict and evaluate on the validation set
y_val_pred = svm_classifier.predict(X_val_tfidf)
print("Validation Set Evaluation:")
print(classification_report(y_val, y_val_pred))

# Predict and evaluate on the test set
y_test_pred = svm_classifier.predict(X_test_tfidf)
print("Test Set Evaluation:")
print(classification_report(y_test, y_test_pred))

# Save confusion matrix for test set
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
pd.DataFrame(test_conf_matrix).to_csv('confusion_matrix_SVM.csv', index=False)
