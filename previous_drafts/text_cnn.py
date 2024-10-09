import pandas as pd
import numpy as np
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten, MaxPooling1D, Input
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

df = pd.read_csv('job_title_industry_draft1.csv', usecols=['WOW Job title', 'WOW Industry', 'New Job Category', 'New Industry Category'])
df.columns

df.dropna(subset=['WOW Job title','New Job Category'], inplace=True)

X = df['WOW Job title']
X = X.apply(lambda x: x.replace(' / ', ','))
X = X.apply(lambda x: x.replace(' /', ','))
X = X.apply(lambda x: x.replace('/', ','))

y = df['New Job Category']

# encoding y
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_hot_encoded = to_categorical(y_encoded)

# splitting data into train and test
x_train, x_test, y_train, y_test = train_test_split(X, y_hot_encoded, test_size=0.2, random_state=42)
# checking unique words in x
results = set()
X.str.lower().str.SPLIT().apply(results.update)
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


model = Sequential()
model.add(Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    trainable=False)
)
model.add(Conv1D(256, 3, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(96, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
batch_size = len(x_train)
history = model.fit(
    x_train, y_train,
    # batch_size=batch_size,
    epochs=300,
    validation_data=(x_test, y_test),
    # callbacks=[checkpoint]
    # callbacks=[early_stopping, checkpoint]
)



# predicting outcomes from original data
# X_tokenized = x_tokenizer.texts_to_sequences(X)
# X_tokenized = pad_sequences(X_tokenized, maxlen=maxlen)
#
results = model.predict(x_test)
predicted_labels_step1 = np.argmax(results, axis=1)
actual_labels_step1 = np.argmax(y_test, axis=1)

comparison_step1 = pd.DataFrame({
    'Predicted Step 1': predicted_labels_step1,
    'Actual Step 1': actual_labels_step1
})

# Display the comparisons side by side
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print("Step 1 (Intermediate Output) Comparison:")
# print(comparison_step1)  # Display first few rows for step 1 comparison

print(len(comparison_step1[comparison_step1['Predicted Step 1'] == comparison_step1['Actual Step 1']])/len(comparison_step1))
print(len(comparison_step1[comparison_step1['Predicted Step 1'] != comparison_step1['Actual Step 1']]))