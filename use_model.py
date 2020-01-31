import pickle

import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('saved_model/cookies_model')

cookie_test = input("Give me a cookie: ")

vectorizer = pickle.load(open("vector.pickel", "rb"))

predict_text = vectorizer.transform([cookie_test])
vocab = vectorizer.vocabulary_
LABELS = ['is a cookie', 'is not a cookie']
predict_array = predict_text.toarray()
prediction = model.predict(predict_array)
print(prediction)
# print(np.argmax(prediction))
print(f'The text {LABELS[np.argmax(prediction)]}')
