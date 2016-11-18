from functools import reduce

from numpy import vstack, hstack, column_stack, asarray
from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics
import nltk



def load_twitter_msgs():
    names = ['id', 'tdate', 'tname', 'ttext', 'ttype', 'trep',
             'tfav', 'tstcount', 'tfol', 'tfrien', 'listcount', 'basename']
    data_negative = read_csv('./data/twitter/negative.csv', delimiter=';', names=names)
    data_positive = read_csv('./data/twitter/positive.csv', delimiter=';', names=names)
    data = vstack((data_negative[['ttext','ttype']], data_positive[['ttext','ttype']]))
    return data


def get_bag_of_words(data, max_features):
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=max_features)
    bag_of_words = vectorizer.fit_transform(data[:, 0]).toarray()
    return bag_of_words, data[:, 1]
    #result = column_stack((bag_of_words, data[:, 1]))
    #result = [(bag_of_words[x], data[x][1]) for x in range(0, bag_of_words.shape[0])]
    #print(result)
    #return result


def get_sequences(data, values):
    return train_test_split(data, values, train_size=0.7, test_size=0.3)
    #return train_test_split(data, values, train_size=0.7, test_size=0.3)

def classification(train_x, train_y, method=BernoulliNB):
    clf = method()
    return clf.fit(train_x, train_y)


def test(clf, test_x, test_y):
    res = clf.predict(test_x)
    return metrics.accuracy_score(test_y, res)

data = load_twitter_msgs()
bag_of_words, values = get_bag_of_words(data, 7000)
x_train, x_test, y_train, y_test = get_sequences(bag_of_words, values)
y_train = asarray(y_train, dtype=int)
y_test = asarray(y_test, dtype=int)
clf = classification(x_train, y_train, MultinomialNB)
res = test(clf, asarray(x_test,dtype=int), asarray(y_test,dtype=int))
print(res)