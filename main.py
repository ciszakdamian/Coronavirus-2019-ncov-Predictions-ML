from model import Model
import pandas as pd

dataset = pd.read_csv('2019-nCoV.csv')

f = dataset.drop(['zgony', 'data'], axis=1)
l = dataset['zgony']

deaths = Model(f, l)

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

    print('Predicted deaths: '+predicted_deaths)
    option = input('\nDo you want repeat prediction? y/n: ')
    if option == 'y' or option == 'Y':
        condition = False
    else:
        condition = True
