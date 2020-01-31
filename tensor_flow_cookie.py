import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import os


def get_data():
    response = requests.get('https://pure-wildwood-95580.herokuapp.com/api/v1/data')

    if response.status_code == 200:
        data = response.json()
        print("Got the data")
    else:
        print("We got nothing")

    for k, v in data.items():
        if k == 'results':
            results = v

    for item in results:
        df = pd.DataFrame.from_dict(results)

    cookies = df[['is_cookie_notice', 'inner_text']]
    cookies['is_cookie_notice'] = cookies['is_cookie_notice'].map({True: 0, False: 1})
    return cookies


def split_data(cookies):
    cookie_text = cookies['inner_text'].values
    cookie_labels = cookies['is_cookie_notice'].values

    cookie_text_train, cookie_text_test, cookie_labels_train, cookie_labels_test = train_test_split(cookie_text,
                                                                                                    cookie_labels,
                                                                                                    test_size=0.25,
                                                                                                    random_state=1000)

    return cookie_text_train, cookie_labels_train, cookie_text_test, cookie_labels_test


def vectorize(cookie_text_train, cookie_text_test):
    vectorizer = CountVectorizer()
    vectorizer.fit(cookie_text_train)

    pickle.dump(vectorizer, open("vector.pickel", "wb"))
    vocab = vectorizer.vocabulary_
    cookies_vec_train = vectorizer.transform(cookie_text_train)
    cookies_vec_test = vectorizer.transform(cookie_text_test)

    return cookies_vec_train, cookies_vec_test, vectorizer, vocab


# print(X_train)
def model_compile(cookies_vect_train_array):
    input_dim = cookie_vec_train_array.shape[1]

    checkpoint_path = 'training_1/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_dim=input_dim, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.save('cookie_model.h5')
    model.save('saved_model/cookies_model')

    return model


def logistic_regression(cookies_vec_train, cookie_labels_train, cookies_vec_test, cookie_labels_test):
    classifier = LogisticRegression()
    classifier.fit(cookies_vec_train, cookie_labels_train)
    score = classifier.score(cookies_vec_test, cookie_labels_test)
    return score


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, train_loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # cookies = get_data()
    # cookie_text_train, cookie_labels_train, cookie_text_test, cookie_labels_test = split_data(cookies)
    #
    # cookies_vec_train, cookies_vec_test, vectorizer, vocab = vectorize(cookie_text_train, cookie_text_test)

    # vectorizer = CountVectorizer()
    # vectorizer.fit(cookie_text_train)
    #
    # cookies_vec_train = vectorizer.transform(cookie_text_train)
    # cookies_vec_test = vectorizer.transform(cookie_text_test)

    # cookie_vec_train_array = cookies_vec_train.toarray()
    # cookies_vec_test_array = cookies_vec_test.toarray()

    # score = logistic_regression(cookies_vec_train, cookie_labels_train, cookies_vec_test, cookie_labels_test)
    # print(f'Accuracy for cookies data by Logistic Regression: {score:.4f}')
    #
    #
    #
    #
    #
    # model = model_compile(cookie_vec_train_array)
    # print(model.summary())
    # checkpoint_path = 'training_1/cp.ckpt'
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
    #                                                  save_weights_only=True,
    #                                                  verbose=1)
    #
    # #
    # history = model.fit(cookie_vec_train_array, cookie_labels_train, epochs=100, verbose=True,
    #                     validation_data=(cookies_vec_test_array, cookie_labels_test), batch_size=10, callbacks=[cp_callback])
    #
    # loss, accuracy = model.evaluate(cookie_vec_train_array, cookie_labels_train, verbose=False)
    # print(f'Training Accuracy: {accuracy}')
    # #
    # loss, accuracy = model.evaluate(cookies_vec_test_array, cookie_labels_test, verbose=False)
    # print(f'Testing Accuracy: {accuracy}')
    #
    # plot_history(history)

    cookie_test = input("Give me a cookie: ")
    LABELS = ['a cookie', 'not a cookie']

    # predict_text = vectorizer.transform([cookie_test])
    # predict_array = predict_text.toarray()
    # print(predict_array)
    # prediction = model.predict(predict_array)
    # print(prediction)
    # print(np.argmax(prediction))
    # print(f'The text is {LABELS[np.argmax(prediction)]}')
    # print(vocab)

    # cookie_text = ["We use cookies to understand how you use our site and to improve your experience. "
    #                "This includes personalizing content and advertising. To learn more, "
    #                "[url=https://yoursite.com/learnmore]click here[/url]. By continuing to use our site, you accept our "
    #                "use of cookies, revised [url=https://yoursite.com/privacy]Privacy Policy[/url] and "
    #                "[url=https://yoursite.com/tos]Terms of Use[/url"]
    # not_cookie_text = [
    #     "his tutorial demonstrates how to generate text using a character-based RNN. We will work with a dataset of"
    #     " Shakespeare's writing from Andrej Karpathy's The Unreasonable Effectiveness of Recurrent Neural Networks. Given a "
    #     "sequence of characters from this data (), train a model to predict the next character in the sequence (). "
    #     "Longer sequences of text can be generated by calling the model repeatedly."]

    # Saving the entire model for tensorflow.js
    #
    # model.save('cookie_model.h5')
    # h5_model = tf.keras.models.load_model('cookie_model.h5')
    #
    # # alternatively the model is saved with SavedModel format that can be used in Tensorflow.js
    #
    # model.save('saved_model/cookies_model')

    # You can load with keras.model.load_model

    new_model = tf.keras.models.load_model('saved_model/cookies_model')  # Can be used in Tensflow.js
    new_model.summary()

    # loss, acc = new_model.evaluate(cookies_vec_test_array, cookie_labels_test, verbose=2)
    # print(f'Saved model, accuracy: {100 * acc:5.2f}%')
    vectorizer = pickle.load(open("vector.pickel", "rb"))

    predict_text = vectorizer.transform([cookie_test])
    predict_array = predict_text.toarray()
    prediction = new_model.predict(predict_array)
    print(prediction)
    print(np.argmax(prediction))
    print(f'The text is {LABELS[np.argmax(prediction)]}')
