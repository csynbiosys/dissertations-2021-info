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

def clean_acquisitions():

    acq = pd.read_csv('../../../data/raw/acquisitions.csv')

    return acq

def clean_ipos():

    ipos = pd.read_csv('../../../data/raw/ipos.csv')

    return ipos

def clean_funding():
    
    funding_cols = ['uuid', 'investment_type', 'announced_on']
    funding = pd.read_csv('../../../data/raw/funding_rounds.csv', usecols=funding_cols)

    # funding = funding.astype({'announced_on': 'str'})
    
    # funding = funding[funding['lead_investor_uuids'].notnull()]
    # No null entries for investor_count after this filter
    
    return funding

def clean_investments():
    inv_cols = ['funding_round_uuid', 'investor_uuid', 'investor_type', 'is_lead_investor']
    inv = pd.read_csv('../../../data/raw/investments.csv', usecols=inv_cols)
    return inv

if __name__=="__main__":
    tabloo.show(clean_funding())

