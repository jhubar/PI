import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import requests # to dowload csv file in github

url = "https://github.com/ADelau/proj0016-epidemic-data/blob/main/data.csv"

# Import datas from github
# Trouver un moyen d'importer le fichier à partir de github

data = pd.read_csv('git_data.csv', sep=',', header=0)
print(data)
