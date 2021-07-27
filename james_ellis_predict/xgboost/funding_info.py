import pandas as pd
import matplotlib.pyplot as plt
from tabloo import show
from datetime import datetime
import numpy as np

# NOTE need to 1 hot the last round investment type

def funding_info(tc, ts, stats=False):
    # CSV imports
    fr_cols = ['uuid', 'investment_type', 'announced_on', 'raised_amount_usd', 'investor_count',
    'org_uuid']
    fr = pd.read_csv('../../../data/raw/funding_rounds.csv', usecols=fr_cols)

    inv_cols = ['funding_round_uuid', 'investor_uuid']
    inv = pd.read_csv('../../../data/raw/investments.csv', usecols=inv_cols)

    # Only interested in funding rounds that occured in the warmup window
    fr = fr[
        (fr['announced_on'] >= tc) &
        (fr['announced_on'] < ts)
        ]

    if stats:
        # Shows significant spikes at the start of each year i.e trust code 4
        fr['announced_on'].value_counts(normalize=True).sort_index().plot()
        plt.show()

    # Removing some empty values
    fr = fr[fr['raised_amount_usd'].notna()]
    fr = fr[ (fr['investor_count'].notna()) & (fr['investor_count']!=0) ]
    

    # Removing series unknown and undisclosed investment rounds
    fr = fr[ (fr['investment_type'] != 'undisclosed') & (fr['investment_type'] != 'series_unknown') ]

    # Round count in the warmup period 
    fr['round_count'] = fr.groupby('org_uuid')['uuid'].transform('size')

    # Total raised amount in all rounds in the warmup period
    fr['total_raised_amount_usd'] = fr.groupby('org_uuid')['raised_amount_usd'].transform('sum')

    # Last round investment type
    fr['last_round_investment_type'] = fr.sort_values('announced_on').groupby('org_uuid')['investment_type'].transform('last')

    # Last round amount raised in usd
    fr['last_round_raised_amount_usd'] = fr.sort_values('announced_on').groupby('org_uuid')['raised_amount_usd'].transform('last')

    # Last round timelapse in months
    def age_months(x):
        fmt = "%Y-%m-%d"
        tdelta = datetime.strptime(ts, fmt) - datetime.strptime(x.tail(1).item(), fmt)
        return round(tdelta.days/365.2425*12)

    fr['last_round_timelapse_months'] = fr.sort_values('announced_on').groupby('org_uuid')['announced_on'].transform(age_months)

    # Create intersection between funding rounds and the investors
    fr = pd.merge(fr, inv, left_on='uuid', right_on='funding_round_uuid', how='left')
    fr = fr.drop(columns = 'funding_round_uuid')
    
    # Number of unique investors in the warmup period
    fr['total_investor_count'] = fr.groupby('org_uuid')['investor_uuid'].transform('nunique')

    # Number of investors in the last funding round in the warmup window
    fr['last_round_investor_count'] = fr.sort_values('announced_on').groupby('org_uuid')['investor_count'].transform('last')

    # Removing any unnecessary columns
    fr = fr.drop(columns = ['uuid','raised_amount_usd', 'investment_type', 'announced_on', 'investor_count', 'investor_uuid'])

    # Dropping duplicates 
    fr = fr.drop_duplicates(subset=['org_uuid'])

    show(fr)

if __name__=="__main__":
    funding_info('2014-12-01','2017-12-01', stats=False)