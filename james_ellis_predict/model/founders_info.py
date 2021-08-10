import pandas as pd
import matplotlib.pyplot as plt
# from tabloo import show
from datetime import datetime
import numpy as np
from utils import splitDataFrameList

# NOTE could add in founders social media presence
# NOTE add in gender 

def founders_info(tc, ts, stats=False):
    # csv import
    peop_cols = ['uuid', 'gender', 'country_code', 'featured_job_organization_uuid', 'featured_job_title']
    peop = pd.read_csv('../../../data/raw/people.csv', usecols=peop_cols)

    deg_cols = ['person_uuid', 'degree_type', 'is_completed']
    deg = pd.read_csv('../../../data/raw/degrees.csv', usecols=deg_cols)

    # Removing na values from people csv
    peop = peop[peop['featured_job_title'].notna()]
    peop = peop[peop['country_code'].notna()]
    peop = peop[ (peop['gender'].notna()) & (peop['gender'] != 'not_provided')]

    # Only interested in founders
    peop = peop[peop['featured_job_title'].str.contains("founder|Founder")]

    # Founders count
    peop['founders_count'] = peop.groupby('featured_job_organization_uuid')['uuid'].transform('size')

    # Male and female count
    peop['founders_male_count'] = peop.groupby('featured_job_organization_uuid')['gender'].transform(lambda x: (x=='male').sum())
    peop['founders_female_count'] = peop.groupby('featured_job_organization_uuid')['gender'].transform(lambda x: (x=='female').sum())

    # Number of different countries for the founders
    peop['founders_dif_country_count'] = peop.groupby('featured_job_organization_uuid')['country_code'].transform('nunique')

    # Removing unfinished and empty degree types
    deg = deg[deg['is_completed'] == True]
    deg = deg[ (deg['degree_type'].notna()) & (deg['degree_type'] != 'unknown') ]
    
    # Importing abbreviations for degree types
    bach = pd.read_csv('degree_abbreviations/bachelor_abv.csv')['Abbreviations']
    mast = pd.read_csv('degree_abbreviations/master_abv.csv')['Abbreviations']
    phd = pd.read_csv('degree_abbreviations/doctorate_abv.csv')['Abbreviations']

    bach_string = '|'.join(bach)
    mast_string = '|'.join(mast)
    phd_string = '|'.join(phd) 

    # Removing white space, periods, and brackets from the degree type field
    to_replace = ['.', ' ', '(', ')']
    for char in to_replace:
        deg['degree_type'] = deg['degree_type'].str.replace(char, "")
    deg['degree_type'] = deg['degree_type'].str.lower()

    # Explode degree types incase person has multiple degrees
    deg = splitDataFrameList(deg, 'degree_type', ',')
    deg = splitDataFrameList(deg, 'degree_type', '&')
    deg = splitDataFrameList(deg, 'degree_type', '/')

    # Strict matching on degree abbreviation types
    # Note, astype(int) converts True False to 1 0 
    deg['has_bachelors'] = deg['degree_type'].str.fullmatch(bach_string, case=False).astype(int)
    deg['has_masters'] = deg['degree_type'].str.fullmatch(mast_string, case=False).astype(int)
    deg['has_phd'] = deg['degree_type'].str.fullmatch(phd_string, case=False).astype(int)

    # Less strict matching criteria for fully written out degree types
    deg['has_bachelors'] = np.where(deg['has_bachelors'] == 0, 
                                    deg['degree_type'].str.contains('bachelor|undergraduate', case=False).astype(int), 
                                    deg['has_bachelors'])
    
    deg['has_masters'] = np.where(deg['has_masters'] == 0, 
                                    deg['degree_type'].str.contains('master', case=False).astype(int), 
                                    deg['has_masters'])
    
    
    # Merge back so multi-degree holders are in the one the row
    deg = deg.groupby('person_uuid').agg({'degree_type':', '.join, 
                                'has_bachelors': 'max', 
                                'has_masters': 'max',
                                'has_phd': 'max'}).reset_index()

    # Drop unnecessary columns
    deg = deg.drop(columns = 'degree_type')

    # Merging degree information to people
    peop = pd.merge(peop, deg, left_on='uuid', right_on='person_uuid', how='left').drop(columns='person_uuid')

    # Replacing NaN for degree information with 0 - treating missing information on degrees as no degree
    degree_columns = ['has_bachelors', 'has_masters', 'has_phd']
    peop[degree_columns] = peop[degree_columns].fillna(0)

    for c in degree_columns:
        peop[c] = peop.groupby('featured_job_organization_uuid')[c].transform('max')

    # Dropping unnecessary columns
    peop = peop.drop_duplicates(subset='featured_job_organization_uuid').drop(columns = ['uuid', 'gender', 'country_code', 'featured_job_title'])

    return peop

if __name__=="__main__":
    founders_info('2014-12-01','2017-12-01', stats=False)