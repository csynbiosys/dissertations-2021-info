'''
Creates the following features:

country_code (need to change this to some form of one hot)
business_sector (1 hot)
age_years: age of the company in years. Arroyo et al. use months but quantisation not right for the trust code
has_email
has_phone
has_facebook_url
has_twitter_url
has_linkedin_url

Optional to add:

description_sentiment
'''

# Module imports
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from tabloo import show

from utils import splitDataFrameList


# NOTE need to add 1 hot encoding for the country variable
# NOTE could add in the description sentiment 

def company_info(tc, ts, stats=False):
    org_cols = ['uuid', 'country_code', 'state_code', 'status', 'total_funding_usd', 'founded_on',
        'facebook_url', 'linkedin_url', 'twitter_url','email', 'phone',  'primary_role', 'category_groups_list'
    ]

    org = pd.read_csv('../../../data/raw/organizations.csv', usecols=org_cols)

    # Only interested in primary roll of company
    org = org[org['primary_role'] == 'company']
    org = org.drop(columns='primary_role')

    # Removing certain rows with no values
    org = org[org['country_code'].notna()]
    org = org[org['total_funding_usd'].notna()]
    org = org[org['founded_on'].notna()] 
    org = org[org['category_groups_list'].notna()] 

    # Only want companies founded between t_c (start of warmup) and t_s (start of simulation)
    org = org[
        (org['founded_on'] >= tc) &
        (org['founded_on'] < ts)
        ]

    if stats:
        # Shows significant spikes at the start of each year i.e trust code 4
        org['founded_on'].value_counts(normalize=True).sort_index().plot()
        plt.show()
    
    # Age of the company at the start of the simulation window 
    # Not sure whether to round up or not

    def age_years(x):
        fmt = "%Y-%m-%d"
        tdelta = datetime.strptime(ts, fmt) - datetime.strptime(x, fmt)
        return round(tdelta.days/365.2425)

    org['age_years'] = org['founded_on'].transform(age_years)
    org = org.drop(columns='founded_on')

    # 1 hot encoding for social media and contact info presence
    cols = ['email', 'phone', 'facebook_url', 'twitter_url', 'linkedin_url']
    for col in cols:
        org['has_{}'.format(col)] = org[col].transform(lambda x: 1 if pd.notna(x) else 0)
    org = org.drop(columns=cols)

    # 1 hot encoding for the category group list

    # Obtains unique categories
    _df = splitDataFrameList(org, 'category_groups_list', ',')
    categories = _df['category_groups_list'].unique()

    for cat in categories:
        org[cat] = org['category_groups_list'].transform(lambda x: 1 if cat in x.split(',') else 0)
    org = org.drop(columns='category_groups_list')


    show(org)

if __name__=="__main__":
    company_info('2014-12-01','2017-12-01', stats=False)
