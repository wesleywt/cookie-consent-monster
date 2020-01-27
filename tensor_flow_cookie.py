import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

response = requests.get('https://pure-wildwood-95580.herokuapp.com/api/v1/data')

if response.status_code == 200:
    data = response.json()
    print("Got the data")
else:
    print("We got nothing")

# The keys of the json file is printed. It shows the keys in the json file.
# It can also be accessed using data["results"].
# But I can't get the items from the results then.

# In[2]:


# We can extract the 'is_cookie_notice' and 'inner_text' from the result which are our labels and features respectively.

# In[3]:


for k, v in data.items():
    if k == 'results':
        results = v

for item in results:
    df = pd.DataFrame.from_dict(results)

cookies = df[['is_cookie_notice', 'inner_text']]

# We can now change the 'is_cookie_notice' to 0 for True and 1 for False to make the modelling easier.
# Modelling requires numerical
# values.

# In[4]:


cookies['is_cookie_notice'] = cookies['is_cookie_notice'].map({True: 0, False: 1})

# ## Splitting the data
# I can split into train and test as well as validation. In the Fuel Efficiency example they do the validation
# split during the
# model fitting.
#
# In[5]:


cookie_text = cookies['inner_text'].values
cookie_labels = cookies['is_cookie_notice'].values
# print(cookie_text)
# print(cookie_labels)

cookie_text_train, cookie_text_test, cookie_labels_train, cookie_labels_test = train_test_split(cookie_text,
                                                                                                cookie_labels,
                                                                                                test_size=0.25,
                                                                                                random_state=1000)
# train, test = train_test_split(cookies, test_size=0.2)
# train, val = train_test_split(train, test_size=0.2)
print(len(cookie_text_train), 'train examples')
# print(len(val), 'validation examples')
print(len(cookie_text_test), 'testing examples')

# ## Formatting the data
# Lets try CountVectorise

# In[6]:


# from nltk.corpus import names
# from nltk.stem import WordNetLemmatizer


# def letters_only(astr):
#     return astr.isalpha()


# all_names = set(names.words())
# lemma = WordNetLemmatizer()


# def clean_text(docs):
#     cleaned_docs = []
#     for doc in docs:
#         cleaned_docs.append(' '.join(
#             [lemma.lemmatize(word.lower()) for word in doc.split() if letters_only(word) and word not in all_names]))
#     return cleaned_docs


# train_cleaning = clean_text(cookie_text_train)
# test_cleaning = clean_text(cookie_text_test)


# vectorizer = CountVectorizer(min_df=0, lowercase=False)
# vectorizer.fit(train_cleaning)
# print(vectorizer.vocabulary_)

# print(vectorizer.transform(cleaned_cookies).toarray())
# cookie_vector_train = vectorizer.transform(cleaned_cookies).toarray()

# print(train.tail())
vectorizer = CountVectorizer()
vectorizer.fit(cookie_text_train)

X_train = vectorizer.transform(cookie_text_train)
X_test = vectorizer.transform(cookie_text_test)

# print(X_train)

classifier = LogisticRegression()
classifier.fit(X_train, cookie_labels_train)
score = classifier.score(X_test, cookie_labels_test)
print(f'Accuracy for cookies data by Logistic Regression: {score:.4f}')

# We can load the data into the dataset format


# In[7]:


# In[9]:


X_train_array = X_train.toarray()
X_test_array = X_test.toarray()

input_dim = X_train_array.shape[1]
print(X_train.shape[1])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_dim=input_dim, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train_array, cookie_labels_train, epochs=100, verbose=True,
                    validation_data=(X_test_array, cookie_labels_test), batch_size=10)

# ## Evaluating the model
#

# In[11]:


loss, accuracy = model.evaluate(X_train_array, cookie_labels_train, verbose=False)
print(f'Training Accuracy: {accuracy}')

loss, accuracy = model.evaluate(X_test_array, cookie_labels_test, verbose=False)
print(f'Testing Accuracy: {accuracy}')


# ## Plotting it
#

# In[12]:


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


plot_history(history)

# ## Prediction
#
#

# In[13]:

cookie_text = ["We use cookies to understand how you use our site and to improve your experience. "
               "This includes personalizing content and advertising. To learn more, "
               "[url=https://yoursite.com/learnmore]click here[/url]. By continuing to use our site, you accept our "
               "use of cookies, revised [url=https://yoursite.com/privacy]Privacy Policy[/url] and "
               "[url=https://yoursite.com/tos]Terms of Use[/url"]
not_cookie_text = [
    "his tutorial demonstrates how to generate text using a character-based RNN. We will work with a dataset of"
    " Shakespeare's writing from Andrej Karpathy's The Unreasonable Effectiveness of Recurrent Neural Networks. Given a "
    "sequence of characters from this data (), train a model to predict the next character in the sequence (). "
    "Longer sequences of text can be generated by calling the model repeatedly."]
LABELS = ['a cookie', 'not a cookie']
predict_text = vectorizer.transform(cookie_text)
predict_array = predict_text.toarray()
prediction = model.predict(predict_array)
print(prediction)
print(np.argmax(prediction))
print(f'The text is {LABELS[np.argmax(prediction)]}')

# Saving the entire model for tensorflow.js

model.save('cookie_model.h5')
h5_model = tf.keras.models.load_model('cookie_model.h5')

# alternatively the model is saved with SavedModel format that can be used in Tensorflow.js

model.save('saved_model/cookies_model')

# You can load with keras.model.load_model

new_model = tf.keras.models.load_model('saved_model/cookies_model')  # Can be used in Tensflow.js
new_model.summary()

loss, acc = new_model.evaluate(X_test_array, cookie_labels_test, verbose=2)
print(f'Saved model, accuracy: {100 * acc:5.2f}%')

predict_text = vectorizer.transform(not_cookie_text)
predict_array = predict_text.toarray()
prediction = new_model.predict(predict_array)
print(prediction)
print(np.argmax(prediction))
print(f'The text is {LABELS[np.argmax(prediction)]}')
