import pandas as pd
from successful_org import successful_org
from data_preprocess import clean_funding
from utils import splitDataFrameList
import tabloo
import numpy as np
from datetime import datetime

def investments(datestring, delta_t=None):
    '''
    Returns: dataframe of successful companies along with their investor
    '''

    sc = successful_org(datestring)
    fund = clean_funding()

    # Only early stage funding
    # invest_type = ['angel', 'series_a', 'seed']
    # fund = fund[fund['investment_type'].isin(invest_type)]

    fund = fund[fund['announced_on'] <= datestring]

    if delta_t:
        fmt = "%Y-%m-%d"
        dt = datetime.strptime(datestring, fmt)
        success_boundary = dt.replace(year=dt.year - int(delta_t))
        success_boundary = success_boundary.strftime(fmt)
        sc = sc[sc['successful_on'] >= success_boundary]

    # Exploding lead_investors column
    fund = splitDataFrameList(fund, 'lead_investor_uuids', ',')

    def most_common(x):
        if len(x.mode()) > 1:
            n = np.random.randint(len(x.mode()))
            return str(x.mode()[n])
        return str(x.mode()[0])

    fund['most_common_invest_type'] = fund.groupby(['lead_investor_uuids'])['investment_type'].transform(most_common)
    df = pd.merge(sc, fund, on='org_uuid', how='inner')

    return df

if __name__=="__main__":
    investments('2020-01-01', delta_t=5)