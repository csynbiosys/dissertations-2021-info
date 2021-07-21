import pandas as pd
from functools import reduce
from data_preprocess import clean_organisations, clean_acquisitions, clean_ipos
import tabloo
import numpy as np

def successful_org(datestring):
    '''
    Parameters:
    datestring (str): 

    Returns:
    Pandas series of successful organisations
    '''

    org = clean_organisations()
    acq = clean_acquisitions()
    ipos = clean_ipos()

    # Remove unnecessary columns
    ipos = ipos[['org_uuid', 'went_public_on']]
    acquirers = acq[['acquirer_uuid', 'acquired_on']]
    acquirees = acq[['acquiree_uuid', 'acquired_on']]

    # Rename for easier merging 
    acquirers = acquirers.rename(columns={'acquirer_uuid':'org_uuid', 'acquired_on': 'made_acquisition_on'})  
    acquirees = acquirees.rename(columns={'acquiree_uuid':'org_uuid'})  

    # Merge dataframes
    data_frames = [ipos, acquirers, acquirees]
    merged_dates = reduce(lambda  left,right: pd.merge(left,right,on=['org_uuid'],
                                            how='outer'), data_frames)

    # Merge on left to only keep org_uuid's in the bipartite graph
    df = pd.merge(org, merged_dates, on=['org_uuid'], how='left')

    cols = ['went_public_on', 'made_acquisition_on', 'acquired_on']
    for col in cols:
        df = df.sort_values(by=['org_uuid', col])
        df = df.drop_duplicates(subset=['org_uuid'], keep='first')


    df = df[
        (df['went_public_on'] <= datestring) |
        (df['made_acquisition_on'] <= datestring) |
        (df['acquired_on'] <= datestring)
    ]

    # CHECK ON HOW MANY HAVE SUCCESSFUL EVENT BEFORE FOUNDED_ON DATE
    # df = df[
    #     (df['went_public_on'] < df['founded_on']) |
    #     (df['made_acquisition_on'] < df['founded_on']) |
    #     (df['acquired_on'] < df['founded_on'])
    # ]
  
    # Replace NaN with a bogus date for easy min function
    df = df.replace(np.NaN, '9999-01-01', regex=True)
    df['successful_on'] = df[['went_public_on', 'made_acquisition_on', 'acquired_on']].min(axis=1)
    
    return df[['org_uuid', 'successful_on']]

if __name__=="__main__":
    successful_org("2013-01-01")


