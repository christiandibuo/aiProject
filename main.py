import numpy
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.utils import shuffle
from Table import Table
import array as arr
def main():
    key = input("press 0 if you want the table of the article as result, press 1 if you want test something else\n")
    '''
        df_train1 = bw.chosingTrainDomain(df_train, 'Birth Control')
        df_test1 = bw.chosingTestDomain(df_test, 'Birth Control')
        ar, X_train, X_test, y_train, y_test = bw.tokenization(df_train1, df_test1)
        ar = shuffle(ar, random_state=0)
        ar = ar[:6]
        mp=[]
        mp.append(ar)
        mt = np.transpose(mp)
        df = pd.DataFrame(mp)
        print(mp)
        with open('table trigram.tex', 'w') as tf:
            tf.write(df.to_latex(label="table unigram"))
    '''
    trainDomain = []
    testDomain = []
    ngram = input('press 1 if you want to use unigrams, press 2 if you want to use bigrams, press 3 if you want to use trigrams\n')
    tf_idf = input('press 1 if you want to use tf_idf, 0 otherwise\n')
    if (key == '0'):
        domain = ['Birth Control', 'Depression', 'Pain', 'Anxiety', 'Diabetes, Type 2']
        t = Table(domain, domain, ngram, tf_idf)
        print(t.table())
    elif (key == '1'):
        domainNumber = int(input("insert the number of domain to train\n"))
        print('train domain')
        for i in range(domainNumber):
            trainDomain.append(input())
        key = input("insert the number of domain to test\n")
        domainNumber = int(key)
        print('test domain')
        for i in range(domainNumber):
            testDomain.append(input())
        t = Table(trainDomain, testDomain, ngram, tf_idf)
        print(t.table())

if __name__ == '__main__':
    main()










