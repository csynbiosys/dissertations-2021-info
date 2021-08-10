# Module imports
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# from tabloo import show

from utils import splitDataFrameList

def company_info(tc, ts, wws_centrality=None, stats=False):
    '''
    Creates the same company information features used by Arroyo et al. for use
    in the warmup period where tc <= warmup period < ts

    Parameters:
    tc (str): Datestring (yyyy-mm-dd) indicating start of the warmup window
    ts (str): Datestring (yyyy-mm-dd) indicating start of the simulation window
    wws_centrality (str): Centrality measure to use for Bonaventura's WWS network. 
                        If None, centrality not included. Must be one of 'betweenness', 
                        'closeness', 'degree', or 'eigenvector'
    
    Returns:
    Pandas dataframe of company information features
    '''

    # Note, total_funding_usd not included here because it could include funding that has taken place during the simulation window.
    # Hence, we calculate total_funding_usd in funding_info where it can be calculated for a specific warmup window
    org_cols = ['uuid', 'country_code', 'founded_on','facebook_url', 'linkedin_url', 'twitter_url',
    'email', 'phone',  'primary_role', 'category_groups_list']

    org = pd.read_csv('../../../data/raw/organizations.csv', usecols=org_cols)
    
    # Only interested in primary roll of company
    org = org[org['primary_role'] == 'company']
    org = org.drop(columns='primary_role')
    
    # Removing certain rows with no values
    org = org[org['country_code'].notna()]
    org = org[org['founded_on'].notna()] 
    org = org[org['category_groups_list'].notna()] 
    
    # Top 9 countries of unicorns, 10th feature as other -reduce number of features given that the vast majority of countries
    # will provide no useful information - dissertation not about geography either - curse of dimensionality
    # https://www.marshall.usc.edu/faculty-research/centers-excellence/center-global-innovation/startup-index-nations-regions
    country_codes = ['USA', 'CHN', 'IND', 'GBR', 'SGP', 'SWE', 'DEU', 'CAN', 'KOR']

    for country in country_codes:
        org[country] = org['country_code'].str.fullmatch(country).astype(int)
    org = org.drop(columns='country_code')

    org['other_country'] = np.where(org[country_codes].sum(axis=1)==0, 1, 0)
    
    # Only want companies founded between tc (start of warmup) and ts (start of simulation)
    org = org[
        (org['founded_on'] >= tc) &
        (org['founded_on'] < ts)
        ]
    
    if stats:
        # Shows significant spikes at the start of each year i.e trust code 4
        org['founded_on'].value_counts(normalize=True).sort_index().plot()
        plt.show()
    
    # Age of the company in years at the start of the simulation window 
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

    # A note for the write up: 88.7% have NaN centrality 
    if wws_centrality: 
        cs = ['betweenness', 'closeness', 'degree', 'eigenvector']
        if wws_centrality not in cs:
            raise ValueError('Invalid centrality measure. Must be one of "betweenness", "closeness", "degree", or "eigenvector"')
        else:
            wws_c_vals = pd.read_csv('network_centralities/wws_network/wws_{}_centralities.csv'.format(wws_centrality))
            org = pd.merge(org, wws_c_vals, left_on='uuid', right_on='org_uuid', how='left').drop(columns=['org_uuid', 'Unnamed: 0'])
            org = org.rename(columns={'{}_centrality'.format(wws_centrality): 'know_how_{}_centrality'.format(wws_centrality)})
    
    return org

if __name__=="__main__":
    show(company_info('2013-12-15','2017-12-15', wws_centrality='closeness', stats=False))
