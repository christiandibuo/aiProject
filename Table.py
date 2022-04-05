from tabulate import tabulate
from BagOfWorld import BagofWorlf
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
import numpy as np

class Table:
    def __init__(self, domain):
        self.domain = domain

    def table3(self):
        bw = BagofWorlf('drugsComTrain.tsv', 'drugsComTest.tsv')
        df_train, df_test = bw.readingFile()


        acc = []
        for i in range (len(self.domain)):
            df_train1 = bw.chosingTrainDomain(df_train, self.domain[i])
            for j in range (len(self.domain)):
                df_test1 = bw.chosingTestDomain(df_test, self.domain[j])
                X_train, X_test, y_train, y_test = bw.tokenization(df_train1, df_test1)
                ppn = Perceptron(max_iter=40, random_state=1)
                ppn.fit(X_train, y_train)
                y_pred = ppn.predict(X_test)
                acc.append(accuracy_score(y_test, y_pred))
        m = []
        while acc !=[]:
            m.append(acc[:len(self.domain)])
            acc = acc[len(self.domain):]




        tab = tabulate(m, headers=self.domain)
        return tab