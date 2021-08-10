import pandas as pd 
from data_preprocess import clean_organisations, clean_jobs

def job_transfer(ts):
    '''
    ts (string): date string in yyyy-mm-dd format for start of simulation window
    '''
    # Data imports
    jobs = clean_jobs()
    org = clean_organisations()

    # Merging founded_on information to the jobs list
    df = pd.merge(jobs, org, left_on='org_uuid', right_on='org_uuid', how='left')
    
    # Only interested in job transfers before the simulation window
    df = df[df['started_on'] < ts]

    # Remove those with no founded on date
    df = df.dropna(subset=['founded_on'])

    # Remove cases where job started before company was founded
    df = df[df['started_on'] >= df['founded_on']]

    # Only keep people who have moved jobs
    df = df[df.groupby('person_uuid').person_uuid.transform(len) > 1]

    # # Remove cases where the employee is only at one organisation, but has changed roles
    df = df.sort_values(by=['person_uuid', 'started_on'])
    cols = ['person_uuid', 'org_uuid']
    df = df.loc[(df[cols].shift() != df[cols]).any(axis=1)]

    return df

if __name__=="__main__":
    job_transfer('2014-12-01')
