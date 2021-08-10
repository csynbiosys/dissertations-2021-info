import pandas as pd
import numpy as np
# from tabloo import show
from sklearn.model_selection import train_test_split

from company_info import company_info
from funding_info import funding_info
from founders_info import founders_info
from target_variable import add_target_variable

def model_data(tc, ts, tf, wws_centrality=None, syn_centrality=None, sample_weight=False, save_path=None):
    '''
    Create a dataframe used as input for the classifier
    
    Parameters:
    tc (str): Datestring (yyyy-mm-dd) indicating start of the warmup window
    ts (str): Datestring (yyyy-mm-dd) indicating start of the simulation window
    tf (str): Datestring (yyyy-mm-dd) indicating end of the simulation window 

    wws_centrality (str): Centrality measure to use for Bonaventura's WWS network. If None, centrality not included
    syn_centrality (str): Centrality measure to use for investor syndicate network. If None, centrality not included
    
    Allowed centralities for both measures include 'betweenness', 'closeness', 'degree', or 'eigenvector'

    sample_weight (False): Boolean to include sample weight for each class. Only used for AUTOML sample_weight 
                            feature and not to be used with the model.
    save_path (str): Optional location and name to save data to. Please include file extension.

    Returns:
    Pandas dataframe of the model data

    '''

    comp = company_info(tc, ts, wws_centrality=wws_centrality)
    fund = funding_info(tc, ts, syn_centrality=syn_centrality)
    founder = founders_info(tc, ts)

    df = pd.merge(comp, fund, left_on='uuid', right_on='org_uuid', how='left').drop(columns='org_uuid')
    df = pd.merge(df, founder, left_on='uuid', right_on='featured_job_organization_uuid', how='left').drop(columns='featured_job_organization_uuid')

    df = add_target_variable(df, tc, ts, tf)
    
    # Removing those that were successful in the warmup period
    df = df[df['successful'] != -1]

    if save_path:
        df.to_csv(save_path, index=False)

    return df
   
if __name__=="__main__":
    dates = ['2013-12-15','2017-12-15', '2020-12-15']
    cs = ['betweenness', 'closeness', 'degree', 'eigenvector']

    # # Baseline
    # model_data(*dates, save_path='model_data/baseline/data.csv')

    # # Baseline + syndicate network
    # for c in cs:
    #     model_data(*dates, syn_centrality=c, save_path='model_data/baseline+syn/data_{}.csv'.format(c))

    # # Baseline + wws network
    # for c in cs:
    #     model_data(*dates, wws_centrality=c, save_path='model_data/baseline+wws/data_{}.csv'.format(c))
    
    # Baseline + wws closeness + syn degree
    model_data(*dates, wws_centrality='closeness', syn_centrality='degree', save_path='model_data/baseline+wws+syn/data_closeness_degree.csv')

    # Baseline + wws closeness + syn eigenvector
    model_data(*dates, wws_centrality='closeness', syn_centrality='eigenvector', save_path='model_data/baseline+wws+syn/data_closeness_eigenvector.csv')