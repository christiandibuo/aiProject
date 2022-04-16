from BagOfWorld import BagofWorlf
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.linear_model import Perceptron
import numpy as np
import pandas as pd

class Table:
    def __init__(self, trainDomain, testDomain, ngram, tf_idf):
        self.trainDomain = trainDomain
        self.testDomain = testDomain
        self.ngram = ngram
        self.tf_idf = tf_idf

    def accuracytable(self):
        bw = BagofWorlf('drugsComTrain.tsv', 'drugsComTest.tsv')
        df_train, df_test = bw.readingFile()
        acc = []
        for i in range(len(self.trainDomain)):
            df_train1 = bw.chosingTrainDomain(df_train, self.trainDomain[i])
            for j in range(len(self.testDomain)):
                df_test1 = bw.chosingTestDomain(df_test, self.testDomain[j])
                X_train, X_test, y_train, y_test = bw.tokenization(df_train1, df_test1, self.ngram, self.tf_idf)
                ppn = Perceptron(random_state=1)
                ppn.fit(X_train, y_train)
                y_pred = ppn.predict(X_test)
                acc.append(accuracy_score(y_test, y_pred))
        m = []
        while acc != []:
            m.append(acc[:(len(self.testDomain))])
            acc = acc[(len(self.testDomain)):]
        mt = np.transpose(m)
        df = pd.DataFrame(data=mt, index=self.testDomain, columns=self.trainDomain)
        df = df.mul(100).round(2).astype(str)
        return df