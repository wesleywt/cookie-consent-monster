import glob
import os

from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np

cookies, labels = [], []

cookies_path = './cookies'
veggies_path = './veggies'
# Open the files in each folder and append the text to the list cookies and label it
for filename in glob.glob(os.path.join(cookies_path, '*.cookie')):
    with open(filename, 'r', encoding="ISO-8859-1") as infile:
        cookies.append(infile.read())
        labels.append(0)

for filename in glob.glob(os.path.join(veggies_path, '*.veggie')):
    with open(filename, 'r', encoding="ISO-8859-1") as infile:
        cookies.append(infile.read())
        labels.append(1)

"""Clean Data: Remove numbers, punctuation and do lemmetization"""


def letters_only(astr):
    return astr.isalpha()


all_names = set(names.words())
lemma = WordNetLemmatizer()


def clean_text(docs):
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(' '.join(
            [lemma.lemmatize(word.lower()) for word in doc.split() if letters_only(word) and word not in all_names]))
    return cleaned_docs


cleaned_cookies = clean_text(cookies)

cv = CountVectorizer(stop_words='english', max_features=500)
term_docs = cv.fit_transform(cleaned_cookies)
feature_mapping = cv.vocabulary_
feature_names = cv.get_feature_names()

"""Train the model"""


def get_label_index(labels):
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index


X_train, X_test, Y_train, Y_test = train_test_split(cleaned_cookies, labels, test_size=0.33, random_state=42)

term_docs_train = cv.fit_transform(X_train)
label_index = get_label_index(Y_train)
term_docs_test = cv.transform(X_test)

clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(term_docs_train, Y_train)
prediction_prob = clf.predict_proba(term_docs_test)
print(prediction_prob[0:10])
prediction = clf.predict(term_docs_test)
print(prediction[:10])
accuracy = clf.score(term_docs_test, Y_test)
print('The accuracy using MultinomialNB is: {0:.1f}%'.format(accuracy * 100))

from sklearn.metrics import confusion_matrix

confusion_matrix(Y_test, prediction, labels=[0, 1])

from sklearn.metrics import precision_score, recall_score, f1_score

print(precision_score(Y_test, prediction, pos_label=1))
print(recall_score(Y_test, prediction, pos_label=1))
print(f1_score(Y_test, prediction, pos_label=1))

print(f1_score(Y_test, prediction, pos_label=0))

from sklearn.metrics import classification_report

report = classification_report(Y_test, prediction)
print(report)

pos_prob = prediction_prob[:, 1]
thresholds = np.arange(0.0, 1.2, 0.1)
true_pos, false_pos = [0] * len(thresholds), [0] * len(thresholds)
for pred, y in zip(pos_prob, Y_test):
    for i, threshold in enumerate(thresholds):
        if pred >= threshold:
            if y == 1:
                true_pos[i] += 1
            else:
                false_pos[i] += 1
        else:
            break

true_pos_rate = [tp / 4 for tp in true_pos]
false_pos_rate = [fp / 6 for fp in false_pos]

import matplotlib.pyplot as plt

plt.figure()
lw = 2
plt.plot(false_pos_rate, true_pos_rate, color='darkorange',
         lw=lw)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import roc_auc_score

print(roc_auc_score(Y_test, pos_prob))
