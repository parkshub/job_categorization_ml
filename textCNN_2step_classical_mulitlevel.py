import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import date
from ast import literal_eval
import h5py
from keras.regularizers import l2
import pickle

# todo multilevel and single level model is pretty much the same, but might make labeling much easier and the more we have the more accurate it might become


class DataPreprocessor:
    def __init__(self, filepath, maxlen=None):
        self.filepath = filepath
        self.maxlen = maxlen
        self.vocab_size = None
        self.embedding_dim = 50
        self.tokenizer = None
        self.embedding_matrix = None
        self.word_index = None
        self.label_encoder_y1 = None  # MultiLabelBinarizer for 'New Job Category'
        self.label_encoder_y2 = None  # LabelEncoder for 'New Industry Category'

    def load_data(self, split=False):
        # df = pd.read_csv(self.filepath,
        #                  usecols=['WOW Job title', 'WOW Industry', 'New Job Category', 'New Industry Category'])

        # df = pd.read_excel(self.filepath, sheet_name='Multi_Andrew_Final_jobtitle+ind', engine='openpyxl')
        df = pd.read_excel(
            "assets/job_title_industry.xlsx",
            sheet_name="Multi_Andrew_Final_jobtitle+ind",
            engine="openpyxl",
        )

        df.dropna(
            subset=["WOW Job title", "New Job Category", "New Industry Category"],
            inplace=True,
        )
        df = df[["WOW Job title", "New Job Category", "New Industry Category"]]

        if split == True:
            df = df[~df["WOW Job title"].str.contains("/")]
            df = df[~df["WOW Job title"].str.contains("&")]
            df = df[~df["WOW Job title"].str.contains(",")]
            df = df[~df["WOW Job title"].str.contains(" and ")]

        df["New Job Category"] = df["New Job Category"].apply(lambda x: x.split(","))
        df["New Job Category"] = df["New Job Category"].apply(
            lambda x: [item.strip() for item in x]
        )
        # df['New Job Category'] = df['New Job Category'].apply(lambda x: literal_eval(x))

        # filtering rows of df with less than 2 rows so we can stratify
        value_counts = df["New Job Category"].value_counts()
        print(value_counts)
        classes_to_remove = value_counts[value_counts < 2].index
        df = df[~df["New Job Category"].isin(classes_to_remove)]

        return df

    def encode_labels(self, df, col_name):
        y_encoded = ""

        if col_name == "New Job Category":
            df[col_name] = df[col_name].apply(
                lambda x: x if isinstance(x, list) else literal_eval(x)
            )
            self.label_encoder_y1 = MultiLabelBinarizer()
            y_encoded = self.label_encoder_y1.fit_transform(df[col_name])

            with open('label_encoder_y1.pkl', 'wb') as f:
                pickle.dump(self.label_encoder_y1, f)

        else:
            # label encoder for single lables
            self.label_encoder_y2 = LabelEncoder()
            y_encoded = self.label_encoder_y2.fit_transform(df[col_name])
            # one-hot encoding
            y_encoded = to_categorical(y_encoded)

            with open('label_encoder_y2.pkl', 'wb') as f:
                pickle.dump(self.label_encoder_y2, f)

        return y_encoded, self.label_encoder_y2

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
        with open(glove_file, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
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
        # model 1
        model1 = Sequential()
        model1.add(
            Embedding(
                input_dim=self.vocab_size,
                output_dim=50,
                weights=[self.embedding_matrix],
                input_length=self.maxlen,
                trainable=False,
            )
        )
        model1.add(Conv1D(128, 3, activation="relu"))
        # model1.add(Conv1D(256, 3, activation='relu'))
        model1.add(GlobalMaxPooling1D())
        model1.add(Dense(64, activation="relu", kernel_regularizer=l2(0.01)))
        # model1.add(Dropout(0.2))
        model1.add(Dropout(0.4))
        model1.add(Dense(y_tr1_shape, activation="sigmoid"))

        # model 2
        model2 = Sequential()
        model2.add(Dense(128, activation="relu"))
        model2.add(Dense(y_tr2_shape, activation="sigmoid"))

        # combining models
        input_layer = Input(shape=(self.maxlen,))
        interm_output = model1(input_layer)
        print("this is iterm output", interm_output.shape)
        final_output = model2(interm_output)

        self.model = Model(inputs=input_layer, outputs=[interm_output, final_output])
        self.model.compile(
            optimizer="adam",
            loss=[
                keras.losses.CategoricalFocalCrossentropy(),
                "categorical_crossentropy",
            ],
            metrics=["accuracy", "accuracy"],
        )
        return self.model

    def train(self, x_tr, y_tr1, y_tr2, x_t, y_t1, y_t2, split=False):
        name_prefix = ["", "split_"]
        today = str(date.today()).replace("-", "_")

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=50, restore_best_weights=True
        )

        checkpoint = ModelCheckpoint(
            f"{name_prefix[split]}textCNN_{today}_multilevel.keras",
            monitor="val_loss",
            save_best_only=True,
        )

        self.model.fit(
            x_tr,
            [y_tr1, y_tr2],
            epochs=50,
            validation_data=(x_t, [y_t1, y_t2]),
            callbacks=[early_stopping, checkpoint],
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
        correct_top_k = [
            1 if y_true[i] in top_k_pred[i] else 0 for i in range(len(y_true))
        ]
        accuracy = sum(correct_top_k) / len(correct_top_k)
        return accuracy


if __name__ == "__main__":
    split = False

    preprocessor = DataPreprocessor(filepath="assets/job_title_industry.xlsx")
    df = preprocessor.load_data()

    y_hot_encoded_model1, label_encoder_model1 = preprocessor.encode_labels(
        df, "New Job Category"
    )
    y_hot_encoded_model2, label_encoder_model2 = preprocessor.encode_labels(
        df, "New Industry Category"
    )

    x_train_raw, x_test_raw, y1_train, y1_test, y2_train, y2_test = train_test_split(
        df["WOW Job title"],
        y_hot_encoded_model1,
        y_hot_encoded_model2,
        test_size=0.2,
        stratify=df["New Job Category"].values,
    )

    print(
        x_train_raw.shape,
        x_test_raw.shape,
        y1_train.shape,
        y2_train.shape,
        y2_test.shape,
    )
    print(x_train_raw)

    # tokenizing and padding
    x_train = preprocessor.tokenize_and_pad(x_train_raw)
    x_test = preprocessor.tokenize_and_pad(x_test_raw)

    # glove embedding
    embedding_matrix = preprocessor.load_glove_embeddings("assets/glove.6B.50d.txt")

    # build and train
    model_builder = JobClassificationModel(
        preprocessor.vocab_size, embedding_matrix, preprocessor.maxlen
    )
    model = model_builder.build_model(y1_train.shape[1], y2_train.shape[1])
    model_builder.train(x_train, y1_train, y2_train, x_test, y1_test, y2_test, split)

    # -------------------------#
    # Evaluating
    # -------------------------#

    # predict
    predictions_step1, predictions_step2 = model.predict(x_test)

    # analysis
    step1_accuracy = Evaluation.accuracy_analysis(predictions_step1, y1_test)
    step2_accuracy = Evaluation.accuracy_analysis(predictions_step2, y2_test)

    print(f"Step 1 Job Category Accuracy: {step1_accuracy}")
    print(f"Step 2 Industry Accuracy: {step2_accuracy}")

    # top 3 accuracy
    top3_accuracy_step1 = Evaluation.top_k_accuracy(
        predictions_step1, np.argmax(y1_test, axis=1), k=3
    )
    print(f"Top-3 Job Category Accuracy: {top3_accuracy_step1}")

    # -------------------------#
    # Predicted outcome top 1, top 3, threshold
    # -------------------------#

    label_mapping_model1 = dict(
        zip(range(len(preprocessor.label_encoder_y1.classes_)), preprocessor.label_encoder_y1.classes_)
    )

    # getting indices for top 1 and 3
    top_1_indices = np.argsort(predictions_step1, axis=1)[:, -1:]
    top_3_indices = np.argsort(predictions_step1, axis=1)[:, -3:]

    predicted_labels_step1_top_1 = np.zeros_like(predictions_step1)
    predicted_labels_step1_top_3 = np.zeros_like(predictions_step1)

    for i, indices in enumerate(top_1_indices):
        predicted_labels_step1_top_1[i, indices] = 1

    for i, indices in enumerate(top_3_indices):
        predicted_labels_step1_top_3[i, indices] = 1

    predicted_labels_step1_top_1_readable = [
        [label_mapping_model1[idx] for idx, value in enumerate(pred) if value == 1]
        for pred in predicted_labels_step1_top_1
    ]

    predicted_labels_step1_top_3_readable = [
        [label_mapping_model1[idx] for idx, value in enumerate(pred) if value == 1]
        for pred in predicted_labels_step1_top_3
    ]

    actual_labels_step1_readable = [
        [label_mapping_model1[idx] for idx, value in enumerate(true) if value == 1]
        for true in y1_test
    ]

    # creating dataframes for top 1 and 3
    step1_df_top_1 = pd.DataFrame(
        {
            "input": x_test_raw,
            "top_1_predicted_labels": predicted_labels_step1_top_1_readable,
            "actual_labels": actual_labels_step1_readable,
        }
    )

    step1_df_top_3 = pd.DataFrame(
        {
            "input": x_test_raw,
            "top_3_predicted_labels": predicted_labels_step1_top_3_readable,
            "actual_labels": actual_labels_step1_readable,
        }
    )

    # instead of top 1 and 3 using threshold, same process
    threshold = 0.5
    predicted_labels_step1 = (predictions_step1 >= threshold).astype(int)

    predicted_labels_step1_readable = [
        [label_mapping_model1[idx] for idx, value in enumerate(pred) if value == 1]
        for pred in predicted_labels_step1
    ]

    actual_labels_step1_readable = [
        [label_mapping_model1[idx] for idx, value in enumerate(true) if value == 1]
        for true in y1_test
    ]

    actual_labels_step2 = np.argmax(y2_test, axis=1)

    step1_df = pd.DataFrame(
        {
            "input": x_test_raw,
            "predicted_labels": predicted_labels_step1_readable,
            "actual_labels": actual_labels_step1_readable,
        }
    )


    # --------------------------------#
    # Step 2 predictions and labels
    # --------------------------------#

    predicted_labels_step2 = np.argmax(predictions_step2, axis=1)

    label_mapping_model2 = dict(
        zip(
            range(len(preprocessor.label_encoder_y2.classes_)),
            preprocessor.label_encoder_y2.classes_,
        )
    )

    predicted_labels_step2_readable = [
        label_mapping_model2[idx] for idx in predicted_labels_step2
    ]

    step2_df = pd.DataFrame(
        {
            "input": x_test_raw,
            "top_1_predicted_label": predicted_labels_step2_readable,
            "actual_label": [
                label_mapping_model2[idx] for idx in np.argmax(y2_test, axis=1)
            ],
        }
    )