import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class BagofWord:
    def __init__(self, trainFilename, testFilename):
        self.trainFile = trainFilename
        self.testFile = testFilename

    def readingFile(self):
        dataSetTrain = pd.read_csv(self.trainFile, delimiter='\t')
        df_train = pd.DataFrame(dataSetTrain)
        dataSetTest = pd.read_csv(self.testFile, delimiter='\t')
        df_test = pd.DataFrame(dataSetTest)
        df_train['rating'] = df_train['rating'].replace([1, 2, 3], -1)
        df_train['rating'] = df_train['rating'].replace([7, 8, 9, 10], 1.0)
        df_train['rating'] = df_train['rating'].replace([4, 5, 6], 0)
        df_test['rating'] = df_test['rating'].replace([1, 2, 3], -1)
        df_test['rating'] = df_test['rating'].replace([7, 8, 9, 10], 1.0)
        df_test['rating'] = df_test['rating'].replace([4, 5, 6], 0)
        return (df_train, df_test)

    def chosingTrainDomain(self, df_train, trainDomain):
        df_train = df_train[df_train["condition"] == trainDomain]
        return df_train

    def chosingTestDomain(self, df_test, testDomain):
        df_test = df_test[df_test["condition"] == testDomain]
        return df_test

    def tokenization(self, df_train, df_test, ngram, tf_idf):

        X_train = df_train.__getattr__('review')
        y_train = df_train.__getattr__('rating')
        X_test = df_test.__getattr__('review')
        y_test = df_test.__getattr__('rating')
        if ngram == '1':
            X_train_counts, X_test_counts = self.unigram(X_train, X_test)
            if tf_idf == '1':
                X_train_counts, X_test_counts = self.Tfidf(X_train_counts, X_test_counts)
            return (X_train_counts, X_test_counts, y_train, y_test)
        if ngram == '2':
            X_train_counts, X_test_counts = self.bigram(X_train, X_test)
            if tf_idf == '1':
                X_train_counts, X_test_counts = self.Tfidf(X_train_counts, X_test_counts)
            return (X_train_counts, X_test_counts, y_train, y_test)
        elif ngram == '3':
            X_train_counts, X_test_counts = self.trigram(X_train, X_test)
            if tf_idf == '1':
                X_train_counts, X_test_counts = self.Tfidf(X_train_counts, X_test_counts)
            return(X_train_counts, X_test_counts, y_train, y_test)

    def unigram(self, X_train, X_test):
        vectorizer = CountVectorizer()
        X_train_counts = vectorizer.fit_transform(X_train)
        X_test_counts = vectorizer.transform(X_test)
        return (X_train_counts, X_test_counts)

    def bigram(self,X_train, X_test):
        vectorizer = CountVectorizer(ngram_range=(1,2))
        X_train_counts = vectorizer.fit_transform(X_train)
        X_test_counts = vectorizer.transform(X_test)
        return (X_train_counts, X_test_counts)

    def trigram(self,X_train, X_test):
        vectorizer = CountVectorizer(ngram_range=(1,3))
        X_train_counts = vectorizer.fit_transform(X_train)
        X_test_counts = vectorizer.transform(X_test)
        return (X_train_counts, X_test_counts)

    def Tfidf(self, X_train, X_test):
        tf_transformer = TfidfTransformer(use_idf=False).fit(X_train)
        X_train_tf = tf_transformer.transform(X_train)
        tf_transformer = TfidfTransformer(use_idf=False).fit(X_test)
        X_test_tf = tf_transformer.transform(X_test)
        return X_train_tf, X_test_tf