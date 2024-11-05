"""
This module uses the saved model, encoder, tokenizer and max length to take in custom inputs
and outputs its predictions.
"""

# pylint: disable=import-error
import sys
import pickle
from datetime import date
import numpy as np

import keras
from keras.models import load_model, Model
from keras.preprocessing.sequence import pad_sequences


class ModelBuilder:
    """
    This class contains all the methods to load in the necessary components of the model
    and the model itself.
    """

    def __init__(self, date):
        self.date = date
        self.tokenizer = None
        self.maxlen = None
        self.label_encoder_y1 = None
        self.label_encoder_y2 = None

    def load_model(self) -> keras.models.Model:
        """
        This function loads in the model.
        """
        model = load_model(
            f"final_model_and_encoders/textCNN_{self.date}_multilevel.keras",
            compile=False,
        )
        model.compile(
            optimizer="adam",
            loss=[
                keras.losses.CategoricalFocalCrossentropy(),
                "categorical_crossentropy",
            ],
            metrics=["accuracy", "accuracy"],
        )

        return model

    def load_pickle(self):
        """
        This function loads in the other components of the model such as the
        tokenizer and label encoder.
        """
        with open(
            f"final_model_and_encoders/input_tokenizer_x_{self.date}.pkl", "rb"
        ) as f:
            self.tokenizer = pickle.load(f)

        with open(
            f"final_model_and_encoders/input_maxlen_x_{self.date}.pkl", "rb"
        ) as f:
            self.maxlen = pickle.load(f)

        with open(
            f"final_model_and_encoders/label_encoder_y1_{self.date}.pkl", "rb"
        ) as f:
            self.label_encoder_y1 = pickle.load(f)

        with open(
            f"final_model_and_encoders/label_encoder_y2_{self.date}.pkl", "rb"
        ) as f:
            self.label_encoder_y2 = pickle.load(f)


def main(user_input: str):
    """
    This function is the main function. It loads the model and its components.
    Using the model, it categorizes the input provided by the user.
    """

    model_builder = ModelBuilder(date="2024_10_29")
    model_builder.load_pickle()
    model = model_builder.load_model()

    sequences = model_builder.tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequences, maxlen=model_builder.maxlen)

    predictions_step1, predictions_step2 = model.predict(padded_sequence)

    label_mapping_model1 = dict(enumerate(model_builder.label_encoder_y1.classes_))
    label_mapping_model2 = dict(enumerate(model_builder.label_encoder_y2.classes_))

    top1_indices_step1 = np.argsort(predictions_step1, axis=1)[:, -1:]
    top3_indices_step1 = np.argsort(predictions_step1, axis=1)[:, -3:]

    top_indices_step2 = np.argmax(predictions_step2, axis=1)

    predicted_labels_step1_top3 = [
        [label_mapping_model1[idx] for idx in top_indices]
        for top_indices in top3_indices_step1
    ]

    predicted_labels_step1_top1 = [
        [label_mapping_model1[idx] for idx in top_indices]
        for top_indices in top1_indices_step1
    ]

    predicted_labels_step2 = [label_mapping_model2[idx] for idx in top_indices_step2]

    print("#----------------RESULTS----------------#")
    print("* Input:", user_input, "\n")
    print("* Job Category")
    print("\t* Top Result:", predicted_labels_step1_top1)
    print("\t* Top 3 Results:", predicted_labels_step1_top3, "\n")
    print("* Industry Category")
    print("\t* Industry: ", predicted_labels_step2)


if __name__ == "__main__":

    if len(sys.argv) > 2:
        print("Please surround string in double quotes if description contains spaces.")
        sys.exit(1)
    elif len(sys.argv) > 1:
        job_desc = sys.argv[1]
        print(f"Input job description: {job_desc}")
    else:
        print("No input provided. Please pass a string as an argument.")
        sys.exit(1)

    main(job_desc)
