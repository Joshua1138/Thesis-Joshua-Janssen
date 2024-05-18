import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

def load_dataset(filepath):
    df = pd.read_csv(filepath, delimiter=';')
    return (df["text"], df["label"])

train_texts, train_labels = load_dataset('train.csv')
test_texts, test_labels = load_dataset('test.csv')
val_texts, val_labels = load_dataset('val.csv')

# Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_texts, test_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_texts, val_labels))

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Create the TextVectorization layer
VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

# Check the first 20 tokens in the vocabulary
vocab = np.array(encoder.get_vocabulary())

print("2 lstm layers")

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10,
                    validation_data=val_dataset,
                    validation_steps=30)

# Predict the test dataset
test_predictions = model.predict(test_dataset)

# Since the model outputs logits, apply sigmoid and then threshold at 0.5 to convert them to class labels
test_predictions = tf.sigmoid(test_predictions)
test_predictions = np.round(test_predictions).astype(int).flatten()

# Generate the classification report
report = classification_report(test_labels, test_predictions, target_names=['Class 0', 'Class 1'])
print(report)

# Generate a confusion matrix
cm = confusion_matrix(test_labels, test_predictions)
cm_df = pd.DataFrame(cm, index=['Class 0', 'Class 1'], columns=['Class 0', 'Class 1'])

# Save the confusion matrix to a CSV file
cm_df.to_csv('confusion_matrix_2lstm.csv', index=True)
print("Confusion matrix saved to confusion_matrix.csv.")
