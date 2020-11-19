import pandas as pd
import numpy as np
def growth_rate(growth, cases):
    return (100*growth/cases)

def __dataProcessing__(self):

    day = np.array(self.dataframe['day'])
    num_positive = np.array(self.dataframe['num_positive_mean'])


    new_day  = []
    new_positive = []
    confirmed_cases_1_day_ago = []
    confirmed_cases_3_day_ago = []
    confirmed_cases_5_day_ago = []
    confirmed_cases_7_day_ago = []

    growth_in_1_day = []
    growth_in_3_day = []
    growth_in_5_day = []
    growth_in_7_day = []

    growth_rate_in_1_day = []
    growth_rate_in_3_day = []
    growth_rate_in_5_day = []
    growth_rate_in_7_day = []

    for i in range(7, len(day)-7):
        new_day.append(day[i])
        new_positive.append(num_positive[i])
        confirmed_cases_1_day_ago.append(num_positive[i-1])
        confirmed_cases_3_day_ago.append(num_positive[i-3])
        confirmed_cases_5_day_ago.append(num_positive[i-5])
        confirmed_cases_7_day_ago.append(num_positive[i-7])

        growth_in_1_day.append(num_positive[i+1])
        growth_in_3_day.append(num_positive[i+3])
        growth_in_5_day.append(num_positive[i+5])
        growth_in_7_day.append(num_positive[i+7])


        growth_rate_in_1_day.append(growth_rate(num_positive[i+1],num_positive[i-1]))
        growth_rate_in_3_day.append(growth_rate(num_positive[i+3],num_positive[i-3]))
        growth_rate_in_5_day.append(growth_rate(num_positive[i+5],num_positive[i-4]))
        growth_rate_in_7_day.append(growth_rate(num_positive[i+7],num_positive[i-7]))



    df_processing = np.vstack((new_day
                    ,new_positive
                    ,confirmed_cases_1_day_ago
                    ,confirmed_cases_3_day_ago
                    ,confirmed_cases_5_day_ago
                    ,confirmed_cases_7_day_ago
                    ,growth_in_1_day
                    ,growth_in_3_day
                    ,growth_in_5_day
                    ,growth_in_7_day
                    ,growth_rate_in_1_day
                    ,growth_rate_in_3_day
                    ,growth_rate_in_5_day
                    ,growth_rate_in_7_day

                    ))
    return pd.DataFrame(df_processing.T, columns=['new_day'
                    ,'new_positive'
                    ,'confirmed_cases_1_day_ago'
                    ,'confirmed_cases_3_day_ago'
                    ,'confirmed_cases_5_day_ago'
                    ,'confirmed_cases_7_day_ago'
                    ,'growth_in_1_day'
                    ,'growth_in_3_day'
                    ,'growth_in_5_day'
                    ,'growth_in_7_day'
                    ,'growth_rate_in_1_day'
                    ,'growth_rate_in_3_day'
                    ,'growth_rate_in_5_day'
                    ,'growth_rate_in_7_day'
])              #13
