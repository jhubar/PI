import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import json
import os
from datetime import datetime



def api_covid(countries = None):


    for i in range(0, len(countries)):

        # if find(name = 'countries'+countries[i]+'.json',path='Data/'):
        #     print("find: {}".format(countries[i]))
        response = requests.get('https://api.covid19api.com/dayone/country/'+str(countries[i]))
        print(response.status_code)
        print(countries[i])
        if response.status_code != 200:
            continue
        data_countries = response.json()
        with open('Data/countries'+countries[i]+'.json', 'w') as json_file:
            json.dump(data_countries, json_file)
        active = []
        deaths =[]
        recovered = []
        confirmed = []
        date = []

        for i in range (0, len(data_countries)):
            active.append(data_countries[i]['Active'])
            deaths.append(data_countries[i]['Deaths'])
            recovered.append(data_countries[i]['Recovered'])
            confirmed.append(data_countries[i]['Confirmed'])
            date.append(datetime.strptime(data_countries[i]['Date'], '%Y-%m-%dT%H:%M:%SZ'))


        plt.plot(date,confirmed, label = 'Confirmed')
        plt.plot(date,deaths, label = 'Deaths ')
        plt.plot(date,recovered, label = 'Active ')
        plt.plot(date,active,label = 'Recovered ' )
    plt.legend()

    plt.show()
    plt.close()

def api_countries():
    response = requests.get('https://api.covid19api.com/countries')
    print(response.status_code)
    data_countries = response.json()
    with open('countries.json', 'w') as json_file:
        json.dump(data_countries, json_file)

    countries = []
    for i in range(0,len(data_countries)):
        countries.append(data_countries[i]['Country'])

    return countries

def load_countries():
    """
    Load countries if the dataset have been loeaded from API
    """
    data_countries = pd.read_json('countries.json')
    countries = np.array(data_countries['Country'])
    return countries


# def plot_unique_country(country = country):
#     data_countries = pd.read_json('countries'+ country + '.json')
#     countries = np.array(data_countries['Country'])
#     active = []
#     deaths =[]
#     recovered = []
#     confirmed = []
#     date = []
#
#     for i in range (0, len(data)):
#         active.append(data[i]['Active'])
#         deaths.append(data[i]['Deaths'])
#         recovered.append(data[i]['Recovered'])
#         confirmed.append(data[i]['Confirmed'])
#         date.append(datetime.strptime(data[i]['Date'], '%Y-%m-%dT%H:%M:%SZ'))
#
#     # print(confirmed)
#     plt.plot(date,confirmed, label = 'Confirmed')
#     plt.plot(date,deaths, label = 'Deaths')
#     plt.plot(date,recovered, label = 'Active')
#     plt.plot(date,active,label = 'Recovered')
#     plt.legend()
#
#     plt.show()
#     plt.savefig('all.png')
#     plt.close()

if __name__ == "__main__":

    # countries = api_countries()
    countries = load_countries()
    api_covid(countries = countries)
    # plot_unique_country(belgium)
