import pandas as pd


class Data:
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)
        self.df = self.dataset.drop(['Last Update', 'Province/State', 'Country/Region', 'Suspected'], axis=1)
        self.df = self.df.fillna(0)

    def features(self):
        dataset = self.df
        features = dataset.drop(['Death'], axis=1)
        return features

    def labels(self):
        dataset = self.df
        labels = dataset['Death']
        return labels

