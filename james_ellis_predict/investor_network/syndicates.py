import pandas as pd
import numpy as np
from datetime import datetime

from data_preprocess import clean_investments, clean_funding
from utils import splitDataFrameList


def syndicates(ts, relation_length = 5):
    '''
    params:
    ts (string):
    relation_length (int):

    Returns: dataframe of successful companies along with their investor
    '''

    inv = clean_investments()
    fr = clean_funding()

    # Add funding round type for descriptive statistics later
    inv = pd.merge(inv, fr, left_on='funding_round_uuid', right_on='uuid', how='inner').drop(columns='uuid')
    
    # Drop solo investors
    inv = inv[inv.groupby('funding_round_uuid').funding_round_uuid.transform(len) > 1]

    # Number of lead investors
    inv['num_lead_investors'] = inv.groupby('funding_round_uuid')['is_lead_investor'].transform(lambda x: sum(x==True))

    # Only want funding rounds within the 5 year window 
    fmt = "%Y-%m-%d"
    ts_dt = datetime.strptime(ts, fmt)
    t_bound = ts_dt.replace(year=ts_dt.year - int(relation_length))
    t_bound = t_bound.strftime(fmt)
    
    inv = inv[ (inv['announced_on'] < ts) & (inv['announced_on'] > t_bound) ] 

    # Although half of the lead_investor fields are empty, we will add directed centralities for completeness
    # print(inv['is_lead_investor'].value_counts(normalize=True, dropna=False))
    inv['is_lead_investor'] = inv['is_lead_investor'].fillna(False)
    
    # Adding investors most common investment type for statistics later
    def most_common(x):
        if len(x.mode()) > 1:
            n = np.random.randint(len(x.mode()))
            return str(x.mode()[n])
        return str(x.mode()[0])

    inv['most_common_invest_type'] = inv.groupby(['investor_uuid'])['investment_type'].transform(most_common)

    return inv

if __name__=="__main__":
    syndicates('2017-12-15', relation_length = 5)