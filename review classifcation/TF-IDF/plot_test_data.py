import os
import nltk
import math
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import tensorflow as tf
from sklearn import svm
from tensorflow import keras
import numpy as np
import data_preprocessing as dp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA


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


positive_files_names = os.listdir("test data/positive")
negative_files_names = os.listdir("test data/negative")

docs_list = []

for file_name in positive_files_names:
    file = open("test data/positive/" + str(file_name), "r")
    docs_list.append(file.read())

for file_name in negative_files_names:
    file = open("test data/negative/" + str(file_name), "r")
    docs_list.append(file.read())

labels = []

tfidf_test_list = []
for review in docs_list:
    tfidf_test_list.append(list(dp.get_tfidf(review).values()))

predictions = model.predict(np.array(tfidf_test_list))

for num in predictions:
    labels.append(int(round(num[0])))
labels[0] = 0

# labels_positive = [1] * len(positive_files_names)
# labels_negative = [0] * len(negative_files_names)

# labels = labels_positive + labels_negative

pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed_matrix = pca.fit_transform(tfidf_test_list)
transformed = pd.DataFrame(transformed_matrix, columns=['X', 'Y'])
transformed["labels"] = labels
# print(transformed)

positive = pd.DataFrame(columns=['X', 'Y', 'labels'])
negative = pd.DataFrame(columns=['X', 'Y', 'labels'])

for index, row in transformed.iterrows():
    if row['labels'] == 1:
        positive = positive.append(pd.DataFrame({"X":[row['X']], "Y":[row['Y']],"labels":[row['labels']]}), ignore_index=True)
    else:
        negative = negative.append(pd.DataFrame({"X":[row['X']], "Y":[row['Y']],"labels":[row['labels']]}), ignore_index=True)


# positive = transformed[:len(positive_files_names)]
# negative = transformed[len(positive_files_names):]
# print(positive)
# print(negative)
# positive.plot.scatter(x='X', y='Y', label='positive', c='blue')
# negative.plot.scatter(x='X', y='Y', label='negative', c='red')

plt.scatter(x=negative['X'], y=negative['Y'], label='negative', c='red')
plt.scatter(x=positive['X'], y=positive['Y'], label='positive', c='blue')


C = 1.0  # SVM regularization parameter
clf = svm.SVC(kernel = 'linear',  gamma=0.7, C=C )
clf.fit(np.array(transformed_matrix), np.array(labels))

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]
plt.plot(xx, yy, 'k-')

plt.legend()
plt.axis([-0.02, 0.02, -0.02, 0.02])
plt.show()

# def plot_svc_decision_boundary(svm_clf, xmin, xmax):
#     w = svm_clf.coef_[0]
#     b = svm_clf.intercept_[0]
#
#     # At the decision boundary, w0*x0 + w1*x1 + b = 0
#     # => x1 = -w0/w1 * x0 - b/w1
#     x0 = np.linspace(xmin, xmax, 200)
#     decision_boundary = -w[0]/w[1] * x0 - b/w[1]
#
#     # margin = 1/w[1]
#     # gutter_up = decision_boundary + margin
#     # gutter_down = decision_boundary - margin
#
#     svs = svm_clf.support_vectors_
#     plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
#     plt.plot(x0, decision_boundary, "k-", linewidth=2)
#     # plt.plot(x0, gutter_up, "k--", linewidth=2)
#     # plt.plot(x0, gutter_down, "k--", linewidth=2)
#
#
#
# svm_clf = svm.SVC(kernel='linear')
# svm_clf.fit(np.array(transformed_matrix), np.array(labels))
#
# plot_svc_decision_boundary(svm_clf, -1, 2.0)
#
# plt.xlabel('feature 1')
# plt.ylabel('feature 2')
# plt.scatter(x=negative['X'], y=negative['Y'], label='negative', c='blue')
# plt.scatter(x=positive['X'], y=positive['Y'], label='positive', c='red')
#
# plt.show()