import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('textCNN_2024_10_02_multilevel.keras')
input_data = "sushi chef"
predictions = model.predict(input_data)
print(predictions)