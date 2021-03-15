import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime

def api_covid(countries = None):

    for i in range(0, len(countries)):
        response = requests.get('https://api.covid19api.com/dayone/country/'+str(countries[i]))
        print(response.status_code)
        print(countries[i])
        data_countries = response.json()
        with open('Data/countries'+countries[i]+'.json', 'w') as json_file:
            json.dump(data_countries, json_file)

def api_countries():
    response = requests.get('https://api.covid19api.com/countries')
    print(response.status_code)
    data_countries = response.json()
    with open('countries.json', 'w') as json_file:
        json.dump(data_countries, json_file)

    countries = []
    for i in range(0,len(data_countries)):
        countries.append(data_countries[i]['Country'])
    print(countries)
    return countries

def load_countries():
    """
    Load countries if the dataset have been loeaded from API
    """
    data_countries = pd.read_json('countries.json')
    countries = np.array(data_countries['Country'])
    return countries


def plot():
    active = []
    deaths =[]
    recovered = []
    confirmed = []
    date = []

    for i in range (0, len(data)):
        active.append(data[i]['Active'])
        deaths.append(data[i]['Deaths'])
        recovered.append(data[i]['Recovered'])
        confirmed.append(data[i]['Confirmed'])
        date.append(datetime.strptime(data[i]['Date'], '%Y-%m-%dT%H:%M:%SZ'))

    # print(confirmed)
    plt.plot(date,confirmed, label = 'Confirmed')
    plt.plot(date,deaths, label = 'Deaths')
    plt.plot(date,recovered, label = 'Active')
    plt.plot(date,active,label = 'Recovered')
    plt.legend()

    plt.show()
    plt.close()

if __name__ == "__main__":

    # countries = api_countries()
    countries = load_countries()
    api_covid(countries = countries)
