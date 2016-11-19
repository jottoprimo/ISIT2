from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import *
from numpy import asarray, row_stack
from pandas import read_csv
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

stop_words = stopwords.words('russian')


def precond(message):
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = snowball.RussianStemmer()
    tokens = tokenizer.tokenize(message.lower())
    tokens = [stemmer.stem(x) for x in tokens if x not in stop_words]
    res = ''
    for t in tokens:
        res += t+' '
    return res


def load_twitter_msgs():
    names = ['id', 'tdate', 'tname', 'ttext', 'ttype', 'trep',
             'tfav', 'tstcount', 'tfol', 'tfrien', 'listcount', 'basename']
    data_negative = read_csv('./data/twitter/negative.csv', delimiter=';', names=names)
    data_positive = read_csv('./data/twitter/positive.csv', delimiter=';', names=names)
    data = row_stack((data_negative[['ttext']], data_positive[['ttext']])).ravel()
    values = row_stack((data_negative[['ttype']], data_positive[['ttype']])).ravel()
    values = [v==1 for v in values]
    data  = asarray(list(map(precond, data)))
    return data, values


def get_bag_of_words(data, max_features):
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=max_features)
    bag_of_words = vectorizer.fit_transform(data).toarray()
    return bag_of_words


def get_sequences(data, values):
    return train_test_split(data, values, train_size=0.7, test_size=0.3)


def classification(train_x, train_y, method=BernoulliNB):
    clf = method()
    return clf.fit(train_x, train_y)


def test(clf, test_x, test_y):
    res = clf.predict(test_x)
    return metrics.accuracy_score(test_y, res)

data, values = load_twitter_msgs()
bag_of_words = get_bag_of_words(data, 3000)
x_train, x_test, y_train, y_test = get_sequences(bag_of_words, values)
y_train = asarray(y_train, dtype=bool)
print(y_train)
y_test = asarray(y_test, dtype=bool)
clf = classification(x_train, y_train,MultinomialNB)
res = test(clf, x_test, y_test)
print(res)