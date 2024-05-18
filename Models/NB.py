# Naive Bayes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

# Load the Dataset with Separator and Split Columns
def load_and_prepare_dataset(file_path):
    df = pd.read_csv(file_path, sep=';', header=None, names=['text', 'label'])
    return df

train_df = load_and_prepare_dataset('train.csv')
test_df = load_and_prepare_dataset('test.csv')
validation_df = load_and_prepare_dataset('val.csv')

# Vectorize the Text Data and Create Model Pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the Naive Bayes Model
model.fit(train_df['text'], train_df['label'])

# Evaluate the model on the validation set
validation_predictions = model.predict(validation_df['text'])
print("Validation Results:")
print("Classification Report:\n", classification_report(validation_df['label'], validation_predictions))

# Evaluate the model on the test set
test_predictions = model.predict(test_df['text'])
print("Test Results:")
print("Classification Report:\n", classification_report(test_df['label'], test_predictions))

# Save the confusion matrix
conf_matrix = confusion_matrix(test_df['label'], test_predictions)
conf_matrix_df = pd.DataFrame(conf_matrix, index=model.classes_, columns=model.classes_)
conf_matrix_df.to_csv('confusion_matrix_NB.csv', index=True)

print("Confusion matrix has been saved to 'confusion_matrix.csv'.")
