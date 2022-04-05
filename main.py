from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from BagOfWorld import BagofWorlf
from Perceptron import Perceptrone
from Table import Table
bw = BagofWorlf('drugsComTrain.tsv', 'drugsComTest.tsv')
df_train, df_test = bw.readingFile()
'''
key = input("press 0 if you want the table of the article as result, press 1 if you want make a personalized test\n")
if key==1:
    print("make the choose")
'''

#print('accuracy value: ', "{:.2%}".format(accuracy_score(y_test, y_pred)),' \ ' , 'cohenn kappa score:' ,"{:.2%}".format(cohen_kappa_score(y_test, y_pred)))

domain = ['Birth Control', 'Depression', 'Pain']
t = Table(domain)
print(t.table3())

'''''
print(confusion_matrix(y_test, y_pred, labels=[1,0,-1], normalize='pred'))
print()
print(confusion_matrix(y_test, y_pred, labels=[1,0,-1], normalize='true'))
print()
print(confusion_matrix(y_test, y_pred, labels=[1,0,-1], normalize='all'))
print()
print(confusion_matrix(y_test, y_pred, labels=[1,0,-1]))
'''











