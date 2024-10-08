import numpy as np
import tensorflow as tf
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import keras


# change date if you want to use different model
date = "2024_10_08"

# Load the saved model
model = tf.keras.models.load_model(f'final_model_and_encoders/textCNN_{date}_multilevel.keras', compile=False)
model.compile(
    optimizer="adam",
    loss=[keras.losses.CategoricalFocalCrossentropy(),"categorical_crossentropy",],
    metrics=["accuracy", "accuracy"]
)

with open(f'final_model_and_encoders/input_tokenizer_x_{date}.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open(f'final_model_and_encoders/input_maxlen_x_{date}.pkl', 'rb') as f:
    maxlen = pickle.load(f)

with open(f'final_model_and_encoders/label_encoder_y1_{date}.pkl', 'rb') as f:
    label_encoder_y1 = pickle.load(f)

with open(f'final_model_and_encoders/label_encoder_y2_{date}.pkl', 'rb') as f:
    label_encoder_y2 = pickle.load(f)


def main():
    # Get user input
    user_input = input("Enter a job title: ")

    sequences = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequences, maxlen=maxlen)

    # Make predictions
    predictions_step1, predictions_step2 = model.predict(padded_sequence)

    label_mapping_model1 = {
        idx: value
        for idx, value in enumerate(label_encoder_y1.classes_)
    }

    label_mapping_model2 = {
        idx: value
        for idx, value in enumerate(label_encoder_y2.classes_)
    }


    top1_indices_step1 = np.argsort(predictions_step1, axis=1)[:, -1:]
    top3_indices_step1 = np.argsort(predictions_step1, axis=1)[:, -3:]

    top_indices_step2 = np.argmax(predictions_step2, axis=1)

    predicted_labels_step1_top3 = [
        [label_mapping_model1[idx] for idx in top_indices] for top_indices in top3_indices_step1
    ]

    predicted_labels_step1_top1 = [
        [label_mapping_model1[idx] for idx in top_indices] for top_indices in top1_indices_step1
    ]

    predicted_labels_step2 = [
        label_mapping_model2[idx] for idx in top_indices_step2
    ]

    print('#----------------RESULTS----------------#')
    print('Input:', user_input, '\n')
    print('Job Category')
    print("\t* Top Result:", predicted_labels_step1_top1)
    print("\t* Top 3 Results:", predicted_labels_step1_top3, '\n')
    print('Industry Category')
    print("\t* Industry: ", predicted_labels_step2)

if __name__ == "__main__":
    main()