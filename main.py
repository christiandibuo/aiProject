from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from BagOfWorld import BagofWorlf



bw = BagofWorlf('drugsComTrain.tsv', 'drugsComTest.tsv')
df_train, df_test = bw.readingFile()

df_train = bw.chosingTrainDomain(df_train, 'Anxiety')
df_test = bw.chosingTestDomain(df_test, 'Anxiety')
X_train_tf, X_test_tf, y_train, y_test = bw.tokenization(df_train, df_test)

ppn = Perceptron(eta0=0.01, max_iter=40, random_state=1)
ppn.fit(X_train_tf, y_train)
y_pred = ppn.predict(X_test_tf)

print('accuracy value: ', "{:.2%}".format(accuracy_score(y_test, y_pred)),' \ ' , 'cohenn kappa score:' ,"{:.2%}".format(cohen_kappa_score(y_test, y_pred)))

'''''
print(confusion_matrix(y_test, y_pred, labels=[1,0,-1], normalize='pred'))
print()
print(confusion_matrix(y_test, y_pred, labels=[1,0,-1], normalize='true'))
print()
print(confusion_matrix(y_test, y_pred, labels=[1,0,-1], normalize='all'))
print()
print(confusion_matrix(y_test, y_pred, labels=[1,0,-1]))
'''











