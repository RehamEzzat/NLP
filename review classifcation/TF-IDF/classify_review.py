import tensorflow as tf
from tensorflow import keras
import numpy as np
import data_preprocessing as dp

review = input("enter review: ")

x = list(dp.get_tfidf(review).values())

model = keras.Sequential([
    keras.layers.Dense(5, input_dim=len(dp.get_distinct_words()), activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(8, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.1),
              loss='binary_crossentropy',
              metrics=['accuracy'])

checkpoint_path = "model_info/cp.ckpt"

model.load_weights(checkpoint_path)
prediction = model.predict(np.array([x]))

for review in prediction[0]:
    if round(review) == 1:
        print("positive")
    else:
        print("negative")

