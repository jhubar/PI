import json

def __saveJson__(data):
    with open('Data/SEIR+.json', 'w') as outfile:
        json.dump(data, outfile)
