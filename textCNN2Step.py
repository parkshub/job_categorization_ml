# todo
#   what advantages and disadvantages of using many filters, same question for kernel size

import pandas as pd
import numpy as np
# from tensorflow.keras.models import Sequential, Model
from keras.models import Sequential, Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten, MaxPooling1D, Input
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

df = pd.read_csv('job_title_industry.csv', usecols=['WOW Job title', 'WOW Industry', 'New Job Category', 'New Industry Category'])
df.columns
df.dropna(subset=['WOW Job title','New Job Category'], inplace=True)

X = df['WOW Job title']
X = X.apply(lambda x: x.replace(' / ', ','))
X = X.apply(lambda x: x.replace(' /', ','))
X = X.apply(lambda x: x.replace('/', ','))

# encoding y for job category
y_model1 = df['New Job Category']
label_encoder_model1 = LabelEncoder()
y_encoded_model1 = label_encoder_model1.fit_transform(y_model1)
y_hot_encoded_model1 = to_categorical(y_encoded_model1)
# encoding y for industry category
y_model2 = df['New Industry Category']
label_encoder_model2 = LabelEncoder()
y_encoded_model2 = label_encoder_model2.fit_transform(y_model2)
y_hot_encoded_model2 = to_categorical(y_encoded_model2)

# splitting data into train and test
x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y_hot_encoded_model1, y_hot_encoded_model2, test_size=0.2, random_state=42)
# checking unique words in x
results = set()
X.str.lower().str.split().apply(results.update)
print(len(results))

# tokenizing
x_tokenizer = Tokenizer(oov_token="<OOV>")
x_tokenizer.fit_on_texts(x_train)
x_train = x_tokenizer.texts_to_sequences(x_train)
x_test = x_tokenizer.texts_to_sequences(x_test)

word_index = x_tokenizer.word_index
maxlen = max(len(seq) for seq in x_train)
word_index = x_tokenizer.word_index
vocab_size = len(word_index) + 1  # Adding 1 because of reserved 0 index
embedding_dim = 50

# padding x
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

word_to_vector = {}
glove_file = 'glove.6B.50d.txt'
embeddings_index = {}

with open(glove_file, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = vector

vocab_size = len(word_index) + 1

embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)  # Get the GloVe vector for the word
    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector


model1 = Sequential()
model1.add(Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    trainable=False)
)
model1.add(Conv1D(256, 3, activation='relu'))
model1.add(GlobalMaxPooling1D())
model1.add(Dense(128, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(96, activation='sigmoid'))

# todo consider adding more data to the input here like resnet
model2 = Sequential()
model2.add(Dense(128, activation='relu'))  # Use an intermediate layer to process output from model1
model2.add(Dense(39, activation='sigmoid'))

input_layer = Input(shape=(18,))
interm_output = model1(input_layer)
final_output = model2(interm_output)

combined_model = Model(inputs=input_layer, outputs=[interm_output, final_output])
combined_model.compile(optimizer='adam', loss=['categorical_crossentropy', 'categorical_crossentropy'], metrics=[['accuracy'], ['accuracy']])
combined_model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
combined_model.fit(
    x_train,
    [y1_train, y2_train],
    epochs=50,
    validation_data=(x_test, [y1_test, y2_test]),
    # callbacks= [checkpoint],
    callbacks= [early_stopping, checkpoint]
)


predictions_step1, predictions_step2 = combined_model.predict(x_test)
predicted_labels_step1 = np.argmax(predictions_step1, axis=1)
predicted_labels_step2 = np.argmax(predictions_step2, axis=1)

actual_labels_step1 = np.argmax(y1_test, axis=1)
actual_labels_step2 = np.argmax(y2_test, axis=1)


#-----------------------------------------------------------------#
# accuracy analysis only considering final output from both models


comparison_step1 = pd.DataFrame({
    'Predicted Step 1': predicted_labels_step1,
    'Actual Step 1': actual_labels_step1
})

comparison_step2 = pd.DataFrame({
    'Predicted Step 2': predicted_labels_step2,
    'Actual Step 2': actual_labels_step2
})


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print("Step 1 (Intermediate Output) Comparison:")
print(len(comparison_step1[comparison_step1['Predicted Step 1'] == comparison_step1['Actual Step 1']])/len(comparison_step1))
print(len(comparison_step1[comparison_step1['Predicted Step 1'] != comparison_step1['Actual Step 1']]))

print("\nStep 2 (Final Output) Comparison:")
print(len(comparison_step2[comparison_step2['Predicted Step 2'] == comparison_step2['Actual Step 2']])/len(comparison_step2))
print(len(comparison_step2[comparison_step2['Predicted Step 2'] != comparison_step2['Actual Step 2']])/len(comparison_step2))


#-----------------------------------------------------------------#
# accuracy analysis considering the top 3 outputs from both models

top3_pred_step1 = np.argsort(predictions_step1, axis=1)[:, -3:]
    # Reverse the order to have the highest first
top3_pred_step1 = np.flip(top3_pred_step1, axis=1)

comparison_step1 = pd.DataFrame({
    'Top 1 Prediction': top3_pred_step1[:, 2],  # Top 1 (largest probability)
    'Top 2 Prediction': top3_pred_step1[:, 1],  # Top 2
    'Top 3 Prediction': top3_pred_step1[:, 0],  # Top 3
    'Actual Label': actual_labels_step1
})

def check_match(row):
    return row['Actual Label'] in [row['Top 1 Prediction'], row['Top 2 Prediction'], row['Top 3 Prediction']]

# Add a new column to indicate if the actual label matches any of the top 3 predictions
comparison_step1['Match'] = comparison_step1.apply(check_match, axis=1)
(comparison_step1['Match'] == True).value_counts()

predicted_labels_step2 = np.argmax(predictions_step2, axis=1)

actual_labels_step1 = np.argmax(y1_test, axis=1)
actual_labels_step2 = np.argmax(y2_test, axis=1)