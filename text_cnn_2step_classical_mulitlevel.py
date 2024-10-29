"""
This module's primary task is to create and train a neural network. As an input, the model takes
in a short job description (4 to 20 words) and categorizes the input into job title and industry.
It contains all the necessary classes and methods to implement training and conduct analysis on
the results.
"""

# pylint: disable=import-error
import json
import pickle
from datetime import date
from ast import literal_eval

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.regularizers import l2
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.losses import CategoricalFocalCrossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input


class DataPreprocessor:
    """
    This class contains all the functions necessary for importing and preprocessing data.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.maxlen = None
        self.vocab_size = None
        self.embedding_dim = 50
        self.tokenizer = None
        self.embedding_matrix = None
        self.word_index = None
        self.label_encoder_y1 = None
        self.label_encoder_y2 = None
        self.today = str(date.today()).replace("-", "_")

    def load_data(self, split_param=False):
        """
        This function loads in the data from an Excel sheet and drops missing rows,
        splits job categories by ',' and removes job categories with less than 2 entries.
        Entries with < 2 where dropped in order to stratify during train_test_split.
        """
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

        if split_param:
            df = df[~df["WOW Job title"].str.contains("/")]
            df = df[~df["WOW Job title"].str.contains("&")]
            df = df[~df["WOW Job title"].str.contains(",")]
            df = df[~df["WOW Job title"].str.contains(" and ")]

        df["New Job Category"] = df["New Job Category"].apply(lambda x: x.split(","))
        df["New Job Category"] = df["New Job Category"].apply(
            lambda x: [item.strip() for item in x]
        )

        value_counts = df["New Job Category"].value_counts()
        classes_to_remove = value_counts[value_counts < 2].index
        df = df[~df["New Job Category"].isin(classes_to_remove)]

        return df

    def encode_labels(self, df: pd.DataFrame, col_name: [str]):
        """
        This function encodes the labels for each model using one-hot-encoding.
        However, model 1 is a multi-label categorization model and can contain more than one 1s.
        Model 1 encoded labels: 0, 0, 1, 0, 1
        Model 2 encoded labels: 0, 0, 1, 0, 0
        """
        if col_name == "New Job Category":
            df[col_name] = df[col_name].apply(
                lambda x: x if isinstance(x, list) else literal_eval(x)
            )
            self.label_encoder_y1 = MultiLabelBinarizer()
            # label_encoder_y = self.label_encoder_y1
            y_encoded = self.label_encoder_y1.fit_transform(df[col_name])

            with open(f"label_encoder_y1_{self.today}.pkl", "wb") as f:
                pickle.dump(self.label_encoder_y1, f)

        else:
            self.label_encoder_y2 = LabelEncoder()
            # label_encoder_y = self.label_encoder_y2
            y_encoded = self.label_encoder_y2.fit_transform(
                df[col_name]
            )  # this transforms to integers
            y_encoded = to_categorical(y_encoded)  # this transforms to one hot encoding

            with open(f"label_encoder_y2_{self.today}.pkl", "wb") as f:
                pickle.dump(self.label_encoder_y2, f)

        return y_encoded, self.label_encoder_y1, self.label_encoder_y2

    def tokenize_and_pad(self, x):
        """
        This function's main purpose is to tokenize and pad the results. If the class doesn't
        already have a tokenizer, it'll assume x belongs to the training set and will fit and
        transform. Otherwise, it'll simply transform using the fitted tokenizer. Due to the
        nature of CNNs, the left most and right most values are often overlooked.
        To compensate, padding was implemented.
        """
        if not self.tokenizer:
            self.tokenizer = Tokenizer(oov_token="<OOV>")
            self.tokenizer.fit_on_texts(x)
            self.word_index = self.tokenizer.word_index

            with open(f"input_tokenizer_x_{self.today}.pkl", "wb") as f:
                pickle.dump(self.tokenizer, f)

        sequences = self.tokenizer.texts_to_sequences(x)

        if not self.maxlen:
            self.maxlen = max(len(seq) for seq in sequences)

            with open(f"input_maxlen_x_{self.today}.pkl", "wb") as f:
                pickle.dump(self.maxlen, f)

        padded_sequences = pad_sequences(sequences, maxlen=self.maxlen)

        self.vocab_size = len(self.word_index) + 1
        return padded_sequences

    def load_glove_embeddings(self, glove_file):
        """
        This function loads pre-trained GloVe embeddings from the famous Stanford set and
        creates an embedding matrix for the model and maps it to the current Tokenizer's vocabulary.
        The embedding used here captures relationships between words.
        For example, "King" - "Man" = "Queen"
        """
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
    """
    This class builds the 2 separate models, concatenates them and trains them.
    """

    def __init__(self, vocab_size, embedding_matrix, maxlen):
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self.maxlen = maxlen
        self.model = None

    def build_model(self, y_tr1_shape, y_tr2_shape):
        """
        This function builds the models, combines them and compiles them.
        """
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
        model1.add(GlobalMaxPooling1D())
        model1.add(Dense(64, activation="relu", kernel_regularizer=l2(0.01)))
        model1.add(Dropout(0.2))
        model1.add(Dense(y_tr1_shape, activation="sigmoid"))

        # model 2
        model2 = Sequential()
        model2.add(Dense(128, activation="relu"))
        model2.add(Dense(y_tr2_shape, activation="sigmoid"))

        # combining models
        input_layer = Input(shape=(self.maxlen,))
        interm_output = model1(input_layer)
        final_output = model2(interm_output)

        self.model = Model(inputs=input_layer, outputs=[interm_output, final_output])
        self.model.compile(
            optimizer="adam",
            loss=[
                CategoricalFocalCrossentropy(),
                "categorical_crossentropy",
            ],
            metrics=["accuracy", "accuracy"],
        )
        return self.model

    def train(self, train_set, test_set, split_param=False):
        """
        This function trains the model. It's set to train for 200 epochs, but stops if validation
        loss doesn't improve for 100 epochs. When training completes, it exports the weights
        that performed the best.
        """

        x_tr, y_tr1, y_tr2 = train_set
        x_t, y_t1, y_t2 = test_set

        name_prefix = ["", "split_"]
        today = str(date.today()).replace("-", "_")

        # early_stopping = EarlyStopping(
        #     monitor="val_loss", patience=100, restore_best_weights=True
        # )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=200, restore_best_weights=True
        )

        checkpoint = ModelCheckpoint(
            f"{name_prefix[split_param]}textCNN_{today}_multilevel.keras",
            monitor="val_loss",
            save_best_only=True,
        )

        self.model.fit(
            x_tr,
            [y_tr1, y_tr2],
            epochs=200,
            validation_data=(x_t, [y_t1, y_t2]),
            callbacks=[early_stopping, checkpoint],
        )

    def evaluate(self, x_t, y_t1, y_t2):
        """
        This function simply evaluates the model.
        """
        results = self.model.evaluate(x_t, [y_t1, y_t2])
        print(f"Evaluation results: {results}")


class Evaluation:
    """
    This class contains a set a static methods to analyze the results of the model.
    """

    @staticmethod
    def accuracy_analysis(predictions, y_true):
        """
        This function calculates the accuracy of the predictions.
        """
        predicted_labels = np.argmax(predictions, axis=1)
        actual_labels = np.argmax(y_true, axis=1)
        correct_predictions = predicted_labels == actual_labels
        accuracy = np.sum(correct_predictions) / len(correct_predictions)
        return accuracy

    @staticmethod
    def f1_analysis(predictions, y_true, average="weighted"):
        """
        This function calculates the F1 score of the predictions.
        """
        predicted_labels = np.argmax(predictions, axis=1)
        actual_labels = np.argmax(y_true, axis=1)
        f1 = f1_score(actual_labels, predicted_labels, average=average)
        return f1

    @staticmethod
    def top_k_accuracy(predictions, y_true, k=3):
        """
        This function calculates the accuracy in terms of whether the model was able to
        guess the correct label within the top three predictions.
        """
        top_k_pred = np.argsort(predictions, axis=1)[:, -k:]
        correct_top_k = [
            1 if y_true[i] in top_k_pred[i] else 0 for i in range(len(y_true))
        ]
        accuracy = sum(correct_top_k) / len(correct_top_k)
        return accuracy

    @staticmethod
    def print_and_write(file, text):
        """
        Prints results to console and writes the results to a .txt file
        """
        print(text)
        file.write(text + "\n")

    @staticmethod
    def output_results(step1, step2):
        """
        This function uses the previous two functions to create a comprehensive analysis
        of how the model performed.
        """
        pred_step1, y1 = step1
        pred_step2, y2 = step2

        step1_accuracy = Evaluation.accuracy_analysis(pred_step1, y1)
        step2_accuracy = Evaluation.accuracy_analysis(pred_step2, y2)

        step1_f1 = Evaluation.f1_analysis(pred_step1, y1)
        step2_f1 = Evaluation.f1_analysis(pred_step2, y2)

        top3_step1_accuracy = Evaluation.top_k_accuracy(
            pred_step1, np.argmax(y1, axis=1), k=3
        )

        today = str(date.today()).replace("-", "_")

        file = open(f"results_{today}.txt", "w", encoding="utf-8")

        Evaluation.print_and_write(
            file, f"Step 1 Job Category Accuracy: {step1_accuracy}"
        )
        Evaluation.print_and_write(file, f"Step 1 F1 Score: {step1_f1}")
        Evaluation.print_and_write(
            file, f"Step 1 Top 3 Job Category Accuracy: {top3_step1_accuracy}"
        )
        Evaluation.print_and_write(file, f"Step 2 Industry Accuracy: {step2_accuracy}")
        Evaluation.print_and_write(file, f"Step 2 F1 Score: {step2_f1}")

    @staticmethod
    def output_readable_results(pred_step1, pred_step2, pp, original_inputs):
        """
        This function converts the input and outputs into a more readable format
        and exports it to a dataframe.
        """
        label_mapping_model1 = dict(enumerate(pp.label_encoder_y1.classes_))

        # getting indices for top 1 and 3
        top1_indices_step1 = np.argsort(pred_step1, axis=1)[:, -1:]
        top3_indices_step1 = np.argsort(pred_step1, axis=1)[:, -3:]

        predicted_labels_step1_top3 = [
            [label_mapping_model1[idx] for idx in top_indices]
            for top_indices in top3_indices_step1
        ]

        predicted_labels_step1_top1 = [
            [label_mapping_model1[idx] for idx in top_indices]
            for top_indices in top1_indices_step1
        ]

        actual_labels_step1 = [
            [label_mapping_model1[idx] for idx, value in enumerate(true) if value == 1]
            for true in y1_test
        ]

        df1 = pd.DataFrame(
            {
                "input": original_inputs,
                "top1_predicted_labels": predicted_labels_step1_top1,
                "top3_predicted_labels": predicted_labels_step1_top3,
                "actual_labels": actual_labels_step1,
            }
        )

        top_indices_step2 = np.argmax(pred_step2, axis=1)

        label_mapping_model2 = dict(enumerate(pp.label_encoder_y2.classes_))

        predicted_labels_step2 = [
            label_mapping_model2[idx] for idx in top_indices_step2
        ]

        df2 = pd.DataFrame(
            {
                "input": original_inputs,
                "predicted_label": predicted_labels_step2,
                "actual_label": [
                    label_mapping_model2[idx] for idx in np.argmax(y2_test, axis=1)
                ],
            }
        )

        today = str(date.today()).replace("-", "_")

        with open(f"step_1_results_{today}.json", "w", encoding="utf-8") as f:
            json.dump(df1.to_json(orient="records"), f)

        with open(f"step_2_results_{today}.json", "w", encoding="utf-8") as f:
            json.dump(df2.to_json(orient="records"), f)

        return df1, df2


if __name__ == "__main__":
    # main()
    SPLIT = False

    # -------------------------#
    # Preprocessing
    # -------------------------#
    preprocessor = DataPreprocessor(filepath="assets/job_title_industry.xlsx")
    job_industry_df = preprocessor.load_data()

    y_hot_encoded_model1, label_encoder_model1, _ = preprocessor.encode_labels(
        job_industry_df, "New Job Category"
    )
    y_hot_encoded_model2, _, label_encoder_model2 = preprocessor.encode_labels(
        job_industry_df, "New Industry Category"
    )

    x_train_raw, x_test_raw, y1_train, y1_test, y2_train, y2_test = train_test_split(
        job_industry_df["WOW Job title"],
        y_hot_encoded_model1,
        y_hot_encoded_model2,
        test_size=0.2,
        stratify=job_industry_df["New Job Category"].values,
    )

    x_train = preprocessor.tokenize_and_pad(x_train_raw)
    x_test = preprocessor.tokenize_and_pad(x_test_raw)

    glove_embedding_matrix = preprocessor.load_glove_embeddings(
        "assets/glove.6B.50d.txt"
    )

    # -------------------------#
    # Building & Training
    # -------------------------#
    model_builder = JobClassificationModel(
        preprocessor.vocab_size, glove_embedding_matrix, preprocessor.maxlen
    )
    model = model_builder.build_model(y1_train.shape[1], y2_train.shape[1])
    model_builder.train(
        [x_train, y1_train, y2_train], [x_test, y1_test, y2_test], SPLIT
    )

    # -------------------------#
    # Evaluating
    # -------------------------#
    predictions_step1, predictions_step2 = model.predict(x_test)

    Evaluation.output_results(
        step1=[predictions_step1, y1_test], step2=[predictions_step2, y2_test]
    )

    step1_df, step2_df = Evaluation.output_readable_results(
        predictions_step1, predictions_step2, preprocessor, x_test_raw
    )
