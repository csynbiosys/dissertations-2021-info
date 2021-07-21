import pandas as pd 
import tabloo
from data_preprocess import clean_organisations, clean_jobs

def remove_duplicates(df):
    
    df = df[df.groupby('person_uuid').person_uuid.transform(len) > 1]

    # # Remove cases where the employee is only at one organisation, but has changed roles
    df = df.sort_values(by=['person_uuid', 'started_on'])
    cols = ['person_uuid', 'org_uuid']
    df = df.loc[(df[cols].shift() != df[cols]).any(axis=1)]

    return df

def job_transfer(year_resolution=False):

    jobs = clean_jobs(year_resolution=year_resolution)
    org = clean_organisations(year_resolution=year_resolution)

    # Merging founded_on information to the jobs list
    jt = pd.merge(jobs, org, left_on='org_uuid', right_on='org_uuid', how='left')
    
    # Remove those with no founded on date
    jt.dropna(subset=['founded_on'], inplace=True)

    # Remove cases where job started before company was founded
    jt = jt[jt['started_on'] >= jt['founded_on']]

    jt = remove_duplicates(jt)

    jt.to_csv('processed_jobs_year_res.csv', index=False)

    return jt

def graph_data(date):

    graph_cols = ['org_uuid', 'person_uuid', 'started_on']
    df = pd.read_csv('processed_jobs_year_res.csv', usecols=graph_cols)
    
    # This in integer comparison - WONT WORK FOR FULL DATES
    df = df[df['started_on'] <= int(date)]

    # print('Number of unique companies: {}'.format(df['org_uuid'].nunique()))
    # print('Number of unique people: {}'.format(df['person_uuid'].nunique()))
    # print('Number of entries in the dataframe: {}'.format(df.shape[0]))

    df = remove_duplicates(df)

    return df


if __name__=="__main__":
    job_transfer(year_resolution=True)
