# Basic data cleaning on the raw files
import pandas as pd 
from datetime import datetime
import time

def clean_organisations():
    # Import data using only the columns needed - takes less time to import
    org_cols = ['uuid', 'founded_on']
    org = pd.read_csv('../../../data/raw/organizations.csv', usecols=org_cols)

    # Renaming UUID to ORG_UUID for convenience
    org = org.rename(columns={'uuid':'org_uuid'})

    return org

def clean_jobs():

    job_cols = ['person_uuid', 'org_uuid', 'started_on', 'ended_on', 'is_current']
    jobs = pd.read_csv('../../../data/raw/jobs.csv', usecols=job_cols)

    # Treats all jobs as equal

    # Remove empty started_on rows
    jobs.dropna(subset=['started_on'], inplace=True)
    
    # NOTE I don't think this is needed if Im not doing any logic with ended_on
    # Remove those who's job is not current, but gave no end date
    # jobs = jobs[(jobs['is_current']==True) | (jobs['ended_on'].notna())]
 
    return jobs

def clean_funding():

    funding = pd.read_csv('../../../data/raw/funding_rounds.csv')

    funding = funding.astype({'announced_on': 'str'})

    return funding

def clean_acquisitions():

    acq = pd.read_csv('../../../data/raw/acquisitions.csv')

    return acq

def clean_ipos():

    ipos = pd.read_csv('../../../data/raw/ipos.csv')

    return ipos

if __name__=="__main__":
    tabloo.show(clean_ipos(year_resolution=False))

