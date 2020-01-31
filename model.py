import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


class Model:
    def __init__(self, features, labels):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(features, labels, test_size=0.1,
                                                                                random_state=0)
        self.model = DecisionTreeClassifier()
        self.model.fit(self.x_train, self.y_train)

    def test(self):
        print('\nModel test: ')
        y_pred = self.model.predict(self.x_test)
        df = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_pred})
        print(df)

# wczytanie danych
dataset = pd.read_csv('2019-nCoV.csv')

# usuniecie zbednych danych i skopiowanie etykiety
f = dataset.drop(['zgony', 'data'], axis=1)
l = dataset['zgony']

deaths = Model(f, l)

deaths.test()
