# Basic data cleaning on the raw files

import pandas as pd 
import tabloo
from datetime import datetime
import time

def clean_organisations(year_resolution=False):
    # Import data using only the columns needed - takes less time to import
    org_cols = ['uuid', 'founded_on']
    org = pd.read_csv('../../../data/raw/organizations.csv', usecols=org_cols)

    # Renaming UUID to ORG_UUID for convenience
    org = org.rename(columns={'uuid':'org_uuid'})
    
    if year_resolution:
        org['founded_on'] = org['founded_on'].str.slice(0,4)

    return org

def clean_acquisitions(year_resolution=False):

    acq = pd.read_csv('../../../data/raw/acquisitions.csv')

    if year_resolution:
        acq['acquired_on'] = acq['acquired_on'].str.slice(0,4)

    return acq

def clean_ipos(year_resolution=False):

    ipos = pd.read_csv('../../../data/raw/ipos.csv')

    if year_resolution:
        ipos['went_public_on'] = ipos['went_public_on'].str.slice(0,4)

    return ipos

def clean_funding(year_resolution=False):

    funding_cols = ['investment_type', 'announced_on', 'investor_count', 'org_uuid', 'lead_investor_uuids']
    funding = pd.read_csv('../../../data/raw/funding_rounds.csv', usecols=funding_cols)

    funding = funding.astype({'announced_on': 'str'})

    if year_resolution:
        funding['announced_on'] = funding['announced_on'].str.slice(0,4)
    
    funding = funding[funding['lead_investor_uuids'].notnull()]
    # No null entries for investor_count after this filter
    
    return funding


if __name__=="__main__":
    tabloo.show(clean_funding())

