from sklearn.linear_model import Perceptron
class Perceptrone:
    def __init__(self, X_train, y_train, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test


    def classical(self, training_parameter, number_epoch):
        ppn = Perceptron(eta0=training_parameter, max_iter=number_epoch, random_state=1)
        ppn.fit(self.X_train, self.y_train)
        y_pred = ppn.predict(self.X_test)
        return y_pred

