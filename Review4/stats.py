import pandas as pd

class stats():

    def __init__(self):
        self.age_path = "stats/age.csv"
        self.households_path = "stats/households.csv"
        self.schools_path = "stats/schools.csv"
        self.workplaces_path = "stats/workplaces.csv"

        self.age_df = None
        self.households_df = None
        self.schools_df = None
        self.workplaces_df = None

    def import_stats(self):
        self.age_df = pd.read_csv(self.age_path)
        self.households_df = pd.read_csv(self.households_path)
        self.schools_df = pd.read_csv(self.schools_path)
        self.workplaces_df = pd.read_csv(self.workplaces_path)




    
