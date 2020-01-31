from model import Model
import pandas as pd


dataset = pd.read_csv('2019-nCoV.csv')

f = dataset.drop(['zgony', 'data'], axis=1)
l = dataset['zgony']


deaths = Model(f, l)
print(deaths.test())
print(deaths.dump('deaths.p'))

model = deaths.load('deaths.p')

infected_people = input('Input how many people is infected: ')
recovered_people = input('Input how many people is recovered: ')
check_data = pd.DataFrame({'zachorowania': [infected_people], 'wyleczeni': [recovered_people]})

print(model.predict(check_data))