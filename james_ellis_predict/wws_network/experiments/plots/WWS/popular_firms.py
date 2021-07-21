import pandas as pd 
import tabloo 
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np

org = pd.read_csv('../../data/raw/organizations.csv', usecols=['uuid', 'name'])
companies = ['Apple','Airbnb', 'Uber', 'Facebook']
popular = org[org['name'].isin(companies)]

dates = [_ for _ in range(1990, 2016)]

for d in dates:
    df = pd.read_csv('../../data/results/centrality_values/bc_{}.csv'.format(d))
    df['rank'] = df['betweenness_centrality'].rank(ascending=False, method='max')

    merge = pd.merge(popular, df, left_on='uuid', right_on='org_uuid', how='left')

    popular['{}'.format(d)] = merge['rank'].values

for comp in companies:  

    v = popular.loc[popular.name == comp, map(str, dates)].values.flatten().tolist()
    
    v_clean = [i for i in v if str(i) != 'nan']
    
    d = dates[len(dates)-len(v_clean):]

    datesnew = np.linspace(d[0], d[-1], 300)  
    spl = make_interp_spline(d, v_clean, k=2)
    v_smooth = spl(datesnew)

    plt.plot(datesnew, v_smooth, label=comp)

plt.legend()
plt.gca().invert_yaxis()
plt.yscale('log')
plt.show()
