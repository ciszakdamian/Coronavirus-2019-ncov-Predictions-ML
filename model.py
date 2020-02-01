import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import os


class Model:
    path = 'models'

    def __init__(self, features, labels):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(features, labels, test_size=0.2,
                                                                                random_state=0)
        self.model = LinearRegression()
        self.model.fit(self.x_train, self.y_train)

    def test(self):
        y_pred = self.model.predict(self.x_test)
        df = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_pred})
        return df

    def dump(self, name):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        pickle.dump(self.model, open(self.path + '/' + name, 'wb'))
        info = 'Model dump to: ' + self.path + '/' + name
        return info

    def load(self, name):
        try:
            m = open(self.path + '/' + name)
            model = pickle.load(open(self.path + '/' + name, 'rb'))
            return model
        except IOError:
            print('Error: Model not exist ' + self.path + '/' + name)
