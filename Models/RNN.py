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

print('datasets loaded in')

# Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_texts, test_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_texts, val_labels))

print('datasets converted')

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print('processed')

# Create the TextVectorization layer
VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

# Check the first 20 tokens in the vocabulary
vocab = np.array(encoder.get_vocabulary())

# Assume `example` is a batch of texts from the dataset
example_texts, example_labels = next(iter(train_dataset))

# Encode these texts
encoded_examples = encoder(example_texts).numpy()

# model
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
print([layer.supports_masking for layer in model.layers])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10,
                    validation_data=val_dataset,
                    validation_steps=30)
test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# Collect all labels and predictions from the test set
all_labels = []
all_predictions = []
for x, y in test_dataset:
    logits = model.predict(x)
    predictions = tf.sigmoid(logits).numpy()
    predictions = (predictions > 0.5).astype(int) 
    all_labels.extend(y.numpy())
    all_predictions.extend(predictions)

# Generate the classification report
report = classification_report(all_labels, all_predictions)
print(report)


# Generate the confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)

# Convert confusion matrix to DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix, 
                              index=['True Neg','True Pos'], 
                              columns=['Pred Neg','Pred Pos'])

# Save the confusion matrix to a CSV file
conf_matrix_df.to_csv('confusion_matrix.csv')
print("Confusion matrix saved to 'confusion_matrix_RNN.csv'.")
