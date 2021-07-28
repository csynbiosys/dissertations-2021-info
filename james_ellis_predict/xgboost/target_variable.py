import pandas as pd
import matplotlib.pyplot as plt
from tabloo import show
import numpy as np

def add_target_variable(org, tc, ts, tf, stats=False):
    '''
    0: unnsuccessful
    1: successful if acquired, ipo, or further funding
    -1: successful but during the warmup period
    '''
    # Loading csv files
    acq_cols = ['acquiree_uuid', 'acquired_on']
    acq = pd.read_csv('../../../data/raw/acquisitions.csv', usecols=acq_cols)
    
    ipo_cols = ['org_uuid', 'went_public_on']
    ipo = pd.read_csv('../../../data/raw/ipos.csv', usecols=ipo_cols)

    fund_cols = ['org_uuid', 'announced_on']
    fund = pd.read_csv('../../../data/raw/funding_rounds.csv', usecols=fund_cols)

    # org = pd.read_csv('../../../data/raw/organizations.csv', usecols=['uuid', 'founded_on'])
    # org = org[ (org['founded_on']>=tc) & (org['founded_on']<ts) ]

    # Checking for IPO or acquisition within the warmup period

    df = pd.merge(org, acq, left_on='uuid', right_on='acquiree_uuid', how='left').drop(columns='acquiree_uuid')
    df = pd.merge(df, ipo, left_on='uuid', right_on='org_uuid', how='left').drop(columns='org_uuid')
    df = pd.merge(df, fund, left_on='uuid', right_on='org_uuid', how='left').drop(columns='org_uuid')

    # If funding date less than the simulation window, replace with NaN, then compress to one company in each row
    df['announced_on'] = np.where(df['announced_on'] < ts, np.NaN, df['announced_on'])
    df = df.sort_values('announced_on').drop_duplicates(subset='uuid')

    # Converting to datetime so dates can be compared to NaN in min operation
    date_cols = ['acquired_on', 'went_public_on', 'announced_on']
    df[date_cols] = df[date_cols].apply(pd.to_datetime)

    # Find first date of successful event in the simulation window
    df['ipo/acq_date'] = df[['acquired_on', 'went_public_on']].min(axis=1)

    conditions = [df['ipo/acq_date'] < ts, 
                ( df['ipo/acq_date'] >= ts ) & ( df['ipo/acq_date'] <= tf ),
                ( df['announced_on'] >= ts ) & ( df['announced_on'] <= tf )]

    choices = [-1, 1, 1]

    df['successful'] = np.select(conditions, choices, default=0)
    
    df = df.drop(columns = ['acquired_on', 'went_public_on', 'announced_on', 'ipo/acq_date'])

    # Acquisitions and IPOs before tc should be naturally filtered out by only considering certain organisations
    if stats:
        acq = acq[acq['acquired_on'] > tc]
        acq['acquired_on'].value_counts(normalize=True).sort_index().plot()
        plt.show()

    return df

if __name__=="__main__":
    target_variable(1,'2013-12-01','2017-12-01', '2020-12-01', stats=False)