# Basic data cleaning on the raw files

import pandas as pd 
import tabloo
from datetime import datetime
import time

def clean_organisations(year_resolution=False):
    # Import data using only the columns needed - takes less time to import
    org_cols = ['uuid', 'founded_on']
    org = pd.read_csv('../data/raw/organizations.csv', usecols=org_cols)

    # Renaming UUID to ORG_UUID for convenience
    org = org.rename(columns={'uuid':'org_uuid'})
    
    if year_resolution:
        org['founded_on'] = org['founded_on'].str.slice(0,4)

    return org

def clean_jobs(year_resolution=False):

    job_cols = ['person_uuid', 'org_uuid', 'started_on', 'ended_on', 'is_current']
    jobs = pd.read_csv('../data/raw/jobs.csv', usecols=job_cols)

    # Treats all jobs as equal

    # Remove empty started_on rows
    jobs.dropna(subset=['started_on'], inplace=True)
    
    # Remove those who's job is not current, but gave no end date
    jobs = jobs[(jobs['is_current']==True) | (jobs['ended_on'].notna())]
    
    if year_resolution:
        jobs['started_on'] = jobs['started_on'].str.slice(0,4)

    return jobs

def clean_funding(year_resolution=False):

    funding = pd.read_csv('../data/raw/funding_rounds.csv')

    funding = funding.astype({'announced_on': 'str'})

    if year_resolution:
        funding['announced_on'] = funding['announced_on'].str.slice(0,4)

    return funding

def clean_acquisitions(year_resolution=False):

    acq = pd.read_csv('../data/raw/acquisitions.csv')

    if year_resolution:
        acq['acquired_on'] = acq['acquired_on'].str.slice(0,4)

    return acq

def clean_ipos(year_resolution=False):

    ipos = pd.read_csv('../data/raw/ipos.csv')

    if year_resolution:
        ipos['went_public_on'] = ipos['went_public_on'].str.slice(0,4)

    return ipos

if __name__=="__main__":
    tabloo.show(clean_ipos(year_resolution=False))

