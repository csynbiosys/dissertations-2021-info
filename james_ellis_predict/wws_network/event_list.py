import pandas as pd 
import tabloo
import numpy as np
from data_preprocess import clean_funding, clean_ipos, clean_acquisitions, clean_organisations
from job_transfer import job_transfer
from functools import reduce
from datetime import datetime

def event_list(year_resolution = False):
    '''
    Creates a csv file containing the dates of major business events: (1) the date of an IPO, (2) the date a company has been acquired,
    (3) the first date a company made an acquisition, (4) the first date of funding, and finally (5) the date the company was founded.
    Should any of events (1)-(4) not occur, values will be left as blank.

    Parameters:
    year_resolution (Boolean): if True, sets all dates to year only. If False, dates will be in yyyy-mm-dd format
    '''

    # Only need companies that are nodes in the bipartite graph
    # NOTE: this might not need called if already saved to a csv file
    org = job_transfer(year_resolution = year_resolution)[['org_uuid', 'founded_on']].drop_duplicates()

    funding = clean_funding(year_resolution = year_resolution)
    acq = clean_acquisitions(year_resolution = year_resolution)
    ipos = clean_ipos(year_resolution = year_resolution)

    # Remove unnecessary columns
    funding = funding[['org_uuid', 'announced_on']]
    ipos = ipos[['org_uuid', 'went_public_on']]
    acquirers = acq[['acquirer_uuid', 'acquired_on']]
    acquirees = acq[['acquiree_uuid', 'acquired_on']]

    # Rename for easier merging 
    acquirers = acquirers.rename(columns={'acquirer_uuid':'org_uuid', 'acquired_on': 'made_acquisition_on'})  
    acquirees = acquirees.rename(columns={'acquiree_uuid':'org_uuid'})  

    # Merge dataframes
    data_frames = [funding, ipos, acquirers, acquirees]
    merged_dates = reduce(lambda  left,right: pd.merge(left,right,on=['org_uuid'],
                                            how='outer'), data_frames)

    # Merge on left to only keep org_uuid's in the bipartite graph
    df = pd.merge(org, merged_dates, on=['org_uuid'], how='left')

    cols = ['announced_on', 'went_public_on', 'made_acquisition_on', 'acquired_on']
    for col in cols:
        df = df.sort_values(by=['org_uuid', col])
        df = df.drop_duplicates(subset=['org_uuid'], keep='first')

    # NOTE there are around 600 cases where events occur before the founded_on date
    # these are naturally put as "unsuccessful" with the delta_t time window. We keep them in the network
   
    df.to_csv('event_list.csv', index=False)
  
    # df = pd.melt(df, id_vars=['org_uuid'], value_vars=['announced_on', 'went_public_on', 'acquired_on'], value_name='event_date')
    # df = df.sort_values(by=['org_uuid', 'event_date'])

    # df.drop_duplicates(subset=['org_uuid'], keep='first', inplace=True)

    # # Adding founded_on dates to the dataframe
    # dates = clean_organisations()[['uuid', 'founded_on']]
    # dates.rename(columns={'uuid':'org_uuid'}, inplace=True)

    # df1 = pd.merge(df, dates, on=['org_uuid'], how='left')

    # df1.to_csv('data/processed/event_list.csv', index=False)


def open_deals(date_string, delta_t = 7, year_resolution=False):
    '''
    Returns a list of open deals at time t, along with a binary success variable if the company is successful between 
    time t and time t + delta_t. 

    A company is defined as an open deal if the company is (1) two years or younger, has (2) not yet received funding, (3) not yet been acquired,
    and (4) not yet been listed in any stock market.

    A successful outcome is defined if atleast one of the following occur: (1) the company makes an acquisition, (2) the 
    company is acquired, or (3) the company undergoes an IPO.

    Parameters:
    date_string (str): string of the date the simulation is to start from
    delta_t (int): number of years allowed for a successful outcome to occur
    '''

    if year_resolution:
        fmt = "%Y"
    else:
        fmt = "%Y-%m-%d"
    
    try:
        dt = datetime.strptime(date_string, fmt)
    except ValueError:
        print("This is the incorrect date string format. It should be YYYY-MM-DD")
    
    # Age boundaries for foundation and event occurence
    age_boundary = dt.replace(year=dt.year - 2)
    age_boundary = age_boundary.strftime(fmt)
    event_boundary = dt.replace(year=dt.year + int(delta_t))
    event_boundary = event_boundary.strftime(fmt)

    # if event_boundary > current_date:
    #     raise 

    event_list = pd.read_csv('event_list.csv').astype(str)
    
    # Filtering for open deals
    # (1) company no older than 2 years
    event_list = event_list[event_list['founded_on'].between(age_boundary, date_string, inclusive=True)]

    # (2) funding not occuring between founded_on and the simulation date
    # NOTE this will also give dates where announced on happened before the founded_on date
    event_list = event_list[~event_list['announced_on'].between(event_list['founded_on'], date_string, inclusive=True)]

    # (3) not yet been acquired
    event_list = event_list[~event_list['acquired_on'].between(event_list['founded_on'], date_string, inclusive=True)]

    # (4) not yet been listed on the stock market
    event_list = event_list[~event_list['went_public_on'].between(event_list['founded_on'], date_string, inclusive=True)]
    
    # Add a column denoting if the open deal is successful within the time window
    conditions = (
        event_list['made_acquisition_on'].between(date_string, event_boundary, inclusive=True) |
        event_list['acquired_on'].between(date_string, event_boundary, inclusive=True) |
        event_list['went_public_on'].between(date_string, event_boundary, inclusive=True)
    )

    event_list['successful'] = np.where(conditions, True, False)

    return event_list[['org_uuid','successful']]
    
if __name__=="__main__":
    dates = [_ for _ in range(1990, 2013)]
    for d in dates:
        print('Year: {}, Random success rate: {}'.format(d, open_deals(str(d), year_resolution=True)['successful'].value_counts(normalize=True)[True]))

  