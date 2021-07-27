import tabloo
import pandas as pd
from data_preprocess import clean_funding, clean_investments

def lead_investor():
    fund = pd.read_csv('../../../data/raw/funding_rounds.csv')
    
    # Some funding rounds have no further information in the investors csv
    # This information is needed to construct the network and hence must be removed
    # Of the 364336 funding rounds, 262467 remain. Hence 101869 had no further information in the investors csv
    investment = clean_investments()['funding_round_uuid'].drop_duplicates()
    print('Number of unique funding rounds in the investor csv: {}'.format(investment.nunique()))
    print('Number of funding rounds in the funding rounds csv: {}'.format(fund.shape[0]))

    merge = pd.merge(investment, fund, left_on='funding_round_uuid', right_on='uuid', how='left')
    print(merge['uuid'].nunique())
    
    print(merge['investor_count'].value_counts(dropna=False))

    tabloo.show(clean_investments())
   

    

    # print(merge['is_lead_investor'].value_counts(normalize=True, dropna=False))

if __name__=="__main__":
    lead_investor()