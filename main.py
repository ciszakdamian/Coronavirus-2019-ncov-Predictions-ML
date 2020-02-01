from model import Model
from data import Data
import pandas as pd

csv = Data('dataset/2019_nCoV_20200121_20200130.csv')
features = csv.features()
labels = csv.labels()

deaths = Model(features, labels)

print("#Model test:")
print(deaths.test())

print('\n'+deaths.dump('deaths.p'))

print("\n#Example Predict:")

model = deaths.load('deaths.p')

condition = False
while not condition:
    infected_people = input('Input - How many people is infected: ')
    recovered_people = input('Input - How many people is recovered: ')

    check = pd.DataFrame({'zachorowania': [infected_people], 'wyleczeni': [recovered_people]})

    predicted_deaths = model.predict(check)
    predicted_deaths = str(predicted_deaths).strip('[]')
    predicted_deaths = str(round(float(predicted_deaths)))

    print('Predicted deaths: '+predicted_deaths)
    option = input('\nDo you want repeat prediction? y/n: ')
    if option == 'y' or option == 'Y':
        condition = False
    else:
        condition = True
