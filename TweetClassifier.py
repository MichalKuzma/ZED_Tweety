import csv
import re
import pandas as pd
import nltk
import string
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from scipy.sparse import csr_matrix

RE_SPACES = re.compile("\s+")
RE_HASHTAG = re.compile("[@#][_a-z0-9]+")
RE_EMOTICONS = re.compile("(:-?\))|(:p)|(:d+)|(:-?\()|(:/)|(;-?\))|(<3)|(=\))|(\)-?:)|(:'\()|(8\))")
RE_HTTP = re.compile("http(s)?://[/\.a-z0-9]+")


class BeforeTokenizationNormalizer:
    @staticmethod
    def normalize(text):
        text = text.strip().lower()
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&amp;', '&')
        text = text.replace('&pound;', u'£')
        text = text.replace('&euro;', u'€')
        text = text.replace('&copy;', u'©')
        text = text.replace('&reg;', u'®')
        return text


class TweetTokenizer:
    @staticmethod
    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        i = 0
        while i < len(tokens):
            token = tokens[i]
            match = re.search(RE_EMOTICONS, token)
            if match is None:
                match = re.search(RE_HASHTAG, token)
            if match is None:
                match = re.search(RE_HTTP, token)

            # sprawdź czy w ramach tokena występuje emotikona, hashtag lub link
            if match is not None:
                foundEmoticon = match.group(0)
                emoticonStart = token.find(foundEmoticon)
                emoticonEnd = emoticonStart + len(foundEmoticon)
                newToken = token[0:emoticonStart] + token[emoticonEnd:]
                tokens.append(match.group(0))
                tokens[i] = newToken
                i -= 1
                # wydziel emotikonę lub hashtag jako token a resztę tekstu rozpatrz ponownie
            else:
                del tokens[i]
                tokens[i:i] = nltk.word_tokenize(token)
            i += 1

        # stwórz stemmer i w pętli stemmuj wszystkie tokeny
        stemmer = nltk.PorterStemmer()
        for i in range(len(tokens)):
            tokens[i] = stemmer.stem(tokens[i])

        return tokens


class TweetClassifier:
    min_word_count = 5

    stopwords = ["a", "about", "after", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been",
                 "before", "being", "between", "both", "by", "could", "did", "do", "does", "doing", "during", "each",
                 "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
                 "him",
                 "himself", "his", "how", "i", "in", "into", "is", "it", "its", "itself", "let", "me", "more", "most",
                 "my",
                 "myself", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "own", "sha",
                 "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
                 "then", "there", "there's", "these", "they", "this", "those", "through", "to", "until", "up", "very",
                 "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "with", "would", "you",
                 "your", "yours", "yourself", "yourselves",
                 "n't", "'s", "'ll", "'re", "'d", "'m", "'ve",
                 "above", "again", "against", "below", "but", "cannot", "down", "few", "if", "no", "nor", "not", "off",
                 "out", "over", "same", "too", "under", "why"]

    def __init__(self, train_file, test_file):
        self.feature_dict = {}
        self.list_of_labels = []
        self.words = Counter()
        self.test_result = {}
        self.classifier = RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=23)
        self.train_tweets = pd.read_csv(train_file, sep=",", na_values=[""],
                                        dtype={"Id": np.str, "Category": np.str, "Tweet": np.str}).dropna()
        self.learn()
        self.test_tweets = pd.read_csv(test_file, sep=",", na_values=[""],
                                       dtype={"Id": np.str, "Tweet": np.str}).dropna()

    def learn(self):
        self.compute_words()

        X_train, y_train = self.create_bow(self.train_tweets, self.feature_dict)
        self.list_of_labels = list(set(y_train))
        self.classifier.fit(X_train, y_train)

    def test(self):
        self.compute_words()

        X_test, y_test = self.create_bow(self.test_tweets, self.feature_dict)
        predicted = self.classifier.predict(X_test)

        self.test_result = {}
        i = 0
        for index, row in self.test_tweets.iterrows():
            self.test_result[row["Id"]] = predicted[i]
            i += 1
        return self.test_result

    def save_test_result_to_file(self, file_name):
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Id', 'Category'])
            for k in self.test_result.keys():
                writer.writerow([k, self.test_result[k]])

    def compute_words(self):
        self.words.clear()
        for index, row in self.train_tweets.iterrows():
            tweet = BeforeTokenizationNormalizer.normalize(row["Tweet"])
            self.words.update(TweetTokenizer.tokenize(tweet))

        for c in string.punctuation:
            del self.words[c]

        for word in self.stopwords:
            del self.words[word]

        common_words = list([k for k, v in self.words.most_common() if v > self.min_word_count])

        self.feature_dict = {}
        for word in common_words:
            self.feature_dict[word] = len(self.feature_dict)

    @staticmethod
    def create_bow(documents, features):
        row = []
        col = []
        data = []

        labels = []

        i = 0
        for _, row_iter in documents.iterrows():
            tweet = BeforeTokenizationNormalizer.normalize(row_iter["Tweet"])
            if "Category" in row_iter:
                label = row_iter["Category"]
                labels.append(label)
            tweet_tokens = TweetTokenizer.tokenize(tweet)

            for token in set(tweet_tokens):
                if token not in features:
                    continue
                row.append(i)
                col.append(features[token])
                data.append(1)
            i += 1
        return csr_matrix((data, (row, col)), shape=(len(documents), len(features))), labels


tweet_classifier = TweetClassifier("train.csv", "test.csv")
print(tweet_classifier.test())
tweet_classifier.save_test_result_to_file("submission.csv")
