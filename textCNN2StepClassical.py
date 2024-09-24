import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import date



class DataPreprocessor:
    def __init__(self, filepath, maxlen=None):
        self.filepath = filepath
        self.maxlen = maxlen
        self.vocab_size = None
        self.embedding_dim = 50
        self.tokenizer = None
        self.embedding_matrix = None
        self.word_index = None
        self.label_encoder_y1 = None
        self.label_encoder_y2 = None

    def load_data(self, split=False):
        # df = pd.read_csv(self.filepath,
        #                  usecols=['WOW Job title', 'WOW Industry', 'New Job Category', 'New Industry Category'])

        df = pd.read_excel(self.filepath, sheet_name='Andrew_Final_jobtitle+industry', engine='openpyxl')


        df.dropna(subset=['WOW Job title', 'New Job Category'], inplace=True)

        if split == True:
            df = df[~df['WOW Job title'].str.contains('/')]
            df = df[~df['WOW Job title'].str.contains('&')]
            df = df[~df['WOW Job title'].str.contains(',')]
            df = df[~df['WOW Job title'].str.contains(' and ')]
            print(len(df))
            return df

        df['WOW Job title'] = df['WOW Job title'].apply(
            lambda x: x.replace(' / ', ',').replace(' /', ',').replace('/', ','))
        return df

    def encode_labels(self, df, col_name):
        if col_name == 'New Job Category':
            self.label_encoder_y1 = LabelEncoder()
            label_encoder = self.label_encoder_y1
        else:
            self.label_encoder_y2 = LabelEncoder()
            label_encoder = self.label_encoder_y2

        # label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(df[col_name])
        y_hot_encoded = to_categorical(y_encoded)
        return y_hot_encoded, label_encoder

    def tokenize_and_pad(self, X):
        if not self.tokenizer:
            self.tokenizer = Tokenizer(oov_token="<OOV>")
            self.tokenizer.fit_on_texts(X)
            self.word_index = self.tokenizer.word_index

        sequences = self.tokenizer.texts_to_sequences(X)

        if not self.maxlen:
            self.maxlen = max(len(seq) for seq in sequences)

        padded_sequences = pad_sequences(sequences, maxlen=self.maxlen)

        self.vocab_size = len(self.word_index) + 1
        return padded_sequences

    def load_glove_embeddings(self, glove_file):
        embeddings_index = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = vector

        self.embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
        return self.embedding_matrix


class JobClassificationModel:
    def __init__(self, vocab_size, embedding_matrix, maxlen):
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self.maxlen = maxlen
        self.model = None

    def build_model(self, y_tr1_shape, y_tr2_shape):
        # Model 1
        model1 = Sequential()
        model1.add(Embedding(
            input_dim=self.vocab_size,
            output_dim=50,
            weights=[self.embedding_matrix],
            input_length=self.maxlen,
            trainable=False)
        )
        model1.add(Conv1D(256, 3, activation='relu'))
        model1.add(GlobalMaxPooling1D())
        model1.add(Dense(128, activation='relu'))
        model1.add(Dropout(0.2))
        # model1.add(Dense(93, activation='sigmoid'))
        model1.add(Dense(y_tr1_shape, activation='sigmoid'))
        # todo change above to x_t1 or x_tr1 shape

        # Model 2
        model2 = Sequential()
        model2.add(Dense(128, activation='relu'))
        model2.add(Dense(y_tr2_shape, activation='sigmoid'))
        # todo change above to y_t2 or y_tr2 shape

        # Combine Models
        print('this is self.maxlen', self.maxlen)
        input_layer = Input(shape=(self.maxlen,))
        interm_output = model1(input_layer)
        print('this is iterm output', interm_output.shape)
        final_output = model2(interm_output)

        self.model = Model(inputs=input_layer, outputs=[interm_output, final_output])
        self.model.compile(optimizer='adam', loss=['categorical_crossentropy', 'categorical_crossentropy'],
                           metrics=['accuracy', 'accuracy'])
        return self.model

    def train(self, x_tr, y_tr1, y_tr2, x_t, y_t1, y_t2, split=False):
        name_prefix = ["", "split_"]
        today = str(date.today()).replace('-', '_')
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        checkpoint = ModelCheckpoint(f'{name_prefix[split]}textCNN_{today}.keras', monitor='val_loss', save_best_only=True)
        self.model.fit(
            x_tr,
            [y_tr1, y_tr2],
            epochs=50,
            validation_data=(x_t, [y_t1, y_t2]),
            callbacks=[early_stopping, checkpoint]
        )

    def evaluate(self, x_t, y_t1, y_t2):
        results = self.model.evaluate(x_t, [y_t1, y_t2])
        print(f"Evaluation results: {results}")


class Evaluation:
    @staticmethod
    def accuracy_analysis(predictions, y_true):
        predicted_labels = np.argmax(predictions, axis=1)
        actual_labels = np.argmax(y_true, axis=1)
        correct_predictions = predicted_labels == actual_labels
        accuracy = np.sum(correct_predictions) / len(correct_predictions)
        return accuracy

    @staticmethod
    def top_k_accuracy(predictions, y_true, k=3):
        top_k_pred = np.argsort(predictions, axis=1)[:, -k:]
        correct_top_k = [1 if y_true[i] in top_k_pred[i] else 0 for i in range(len(y_true))]
        accuracy = sum(correct_top_k) / len(correct_top_k)
        return accuracy


if __name__ == "__main__":
    split = False
    # Data Preprocessing
    # preprocessor = DataPreprocessor(filepath='job_title_industry.csv')
    preprocessor = DataPreprocessor(filepath='job_title_industry.xlsx')
    df = preprocessor.load_data(split=split)
    # df = preprocessor.load_data()

    y_hot_encoded_model1, label_encoder_model1 = preprocessor.encode_labels(df, 'New Job Category')
    y_hot_encoded_model2, label_encoder_model2 = preprocessor.encode_labels(df, 'New Industry Category')

    # Splitting data into train and test
    x_train_raw, x_test_raw, y1_train, y1_test, y2_train, y2_test = train_test_split(df['WOW Job title'],
                                                                                     y_hot_encoded_model1,
                                                                                     y_hot_encoded_model2,
                                                                                     test_size=0.2, random_state=42)

    # Tokenizing and padding
    x_train = preprocessor.tokenize_and_pad(x_train_raw)
    x_test = preprocessor.tokenize_and_pad(x_test_raw)

    # Loading GloVe embeddings
    embedding_matrix = preprocessor.load_glove_embeddings('glove.6B.50d.txt')

    # Building and training the model
    model_builder = JobClassificationModel(preprocessor.vocab_size, embedding_matrix, preprocessor.maxlen)
    model = model_builder.build_model(y1_train.shape[1], y2_train.shape[1])
    model_builder.train(x_train, y1_train, y2_train, x_test, y1_test, y2_test, split)

    # Evaluate the model
    predictions_step1, predictions_step2 = model.predict(x_test)

    # Accuracy analysis
    step1_accuracy = Evaluation.accuracy_analysis(predictions_step1, y1_test)
    step2_accuracy = Evaluation.accuracy_analysis(predictions_step2, y2_test)

    print(f'Step 1 Job Category Accuracy: {step1_accuracy}')
    print(f'Step 2 Industry Accuracy: {step2_accuracy}')

    # Top-3 Accuracy for step1 predictions
    top3_accuracy_step1 = Evaluation.top_k_accuracy(predictions_step1, np.argmax(y1_test, axis=1), k=3)
    print(f'Top-3 Job Category Accuracy: {top3_accuracy_step1}')


    # todo here
    label_mapping_model1 = dict(zip(range(len(label_encoder_model1.classes_)), label_encoder_model1.classes_))

    predicted_labels_step1 = np.argmax(predictions_step1, axis=1)
    predicted_labels_step2 = np.argmax(predictions_step2, axis=1)

    actual_labels_step1 = np.argmax(y1_test, axis=1)
    actual_labels_step2 = np.argmax(y2_test, axis=1)

    step1_df = pd.DataFrame({
        "input": x_test_raw,
        "pred": predicted_labels_step1,
        "actual": actual_labels_step1
    })

    step1_df[['pred', 'actual']] = step1_df[['pred', 'actual']].replace(label_mapping_model1)
    step1_df_incorrect = step1_df[step1_df["pred"] != step1_df['actual']].sort_values(by="pred")
    step1_df_incorrect_counts = step1_df_incorrect[['pred', 'actual']].value_counts()
    step1_df_incorrect_counts = step1_df_incorrect_counts.reset_index()