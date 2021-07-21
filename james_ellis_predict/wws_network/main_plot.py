import pandas as pd 
from event_list import open_deals
import tabloo

dates = [_ for _ in range(2000,2010)]

for d in dates:

    df = pd.read_csv('data/results/centrality_values/cc_{}.csv'.format(d))
    od = open_deals(str(d), year_resolution=True)

    df = pd.merge(df, od, on=['org_uuid'], how='right')

    df = df.sort_values(['betweenness_centrality'], ascending=False)

    print('Year: {}, Accuracy: {}'.format(d, df.head(20)['successful'].value_counts(normalize=True)[True]))
