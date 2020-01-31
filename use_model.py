import tensorflow as tf
import tensorflow_hub as hub
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle
import tensorflow as tf
import os
from keras.models import load_model

tf.keras.backend.clear_session()
tf.random.set_seed(0)
np.random.seed(0)

model = tf.keras.models.load_model('saved_model/cookies_model')

# print(model.summary())

text = [
    'Privacy Notice: Our site uses cookies for advertising, analytics and to improve our sites and services. By continuing to use our site, you agree to our cookies. For ore information, including how to change your settings, see our ']

vectorizer = pickle.load(open("vector.pickel", "rb"))

#


predict_text = vectorizer.transform(text)
vocab = vectorizer.vocabulary_
LABELS = ['is a cookie', 'is not a cookie']
predict_array = predict_text.toarray()
prediction = model.predict(predict_array)
print(prediction)
print(np.argmax(prediction))
print(f'The text  {LABELS[np.argmax(prediction)]}')
print(vocab)

# #Load H5 model
# def load_H5_model () :
#     model = tf.keras.models.load_model('cookie_model.h5')
#     print(model.summary())
#     return model
#
# # def UniversalEmbedding(x):
# #     embedding = embed(x)
# #     return embedding
#
# # Format the incoming text
# def prepare_text(text):
#     vect_fit = pickle.load(open('feature.pkl', 'rb'))
#     vectorizer = CountVectorizer(vocabulary=vect_fit.vocabulary_)
#     tensor_text = vectorizer.transform(text)
#     tensor_text = tensor_text.toarray()
#
#
#
#     return tensor_text
# # Predict what the text is:
# def prediction(model, tensor_text):
#     LABEL = ['is a cookie', 'not a cookie']
#     predict = model.predict([tensor_text])
#     print(predict)
#     decision = LABEL[np.argmax(predict)]
#     return decision
#
# if __name__ == '__main__':
#     model = tf.keras.models.load_model('saved_model/cookies_model')
#
#     text = input("Load text to detect cookie: ")
#     with open("feature.pkl", "rb") as f:
#         vect_fit = pickle.load(f)
#     vectorizer = CountVectorizer(vocabulary=vect_fit.vocabulary_)
#     predict_text = vectorizer.transform([text])
#     print(prepare_text)
#     predict_array = predict_text.toarray()
#     print(predict_array)
#     prediction = model.predict(predict_array)
#     LABELS = ['is a cookie', 'is not a cookie']
#     print(prediction)
#     print(np.argmax(prediction))
#     print(f'The text is {LABELS[np.argmax(prediction)]}')
