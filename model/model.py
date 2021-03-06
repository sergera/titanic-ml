import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import accuracy_score

class Model:
    def __init__(self, test_size=0.1, seed=0, solver='saga', max_iter=100):
        self.model = LogisticRegression(random_state=seed, solver=solver, max_iter=max_iter)

    def fit(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model.fit(self.X_train, self.y_train)

    def predict(self, data):
        ''' obtain predictions'''
        predictions = self.model.predict(data)
        return predictions
    
    def evaluate_model(self):
        '''evaluates trained model on train and test sets'''
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)

        print('train accuracy: ',accuracy_score(self.y_train, pd.Series(train_pred)))
        print('test accuracy: ',accuracy_score(self.y_test, pd.Series(test_pred)))

        print('train r2: {}'.format(r2_score(self.y_train, train_pred)))
        print('test r2: {}'.format(r2_score(self.y_test, test_pred)))