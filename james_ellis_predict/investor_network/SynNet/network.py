import networkx as nx 
import pandas as pd
from investments import investments
from networkx.algorithms import bipartite
from networkx.algorithms.centrality import closeness_centrality, degree_centrality, betweenness_centrality
import tabloo
from utils import from_pandas_edgelist_weighted

def centrality_t(datestring, metric = 'closeness', delta_t = None, syndication = False):
    '''
    Creates a network at a given time 
    '''

    metrics = {
        'closeness': closeness_centrality,
        'degree': degree_centrality,
        'betweenness': betweenness_centrality
    }

    if metric not in metrics:
        raise Exception("Invalid centrality metric. Pick one of 'closeness', 'degree', 'betweenness'")

    data = investments(datestring, delta_t = delta_t)
    
    other_data = data[['lead_investor_uuids', 'most_common_invest_type']].drop_duplicates(subset='lead_investor_uuids').rename(columns={'lead_investor_uuids':'uuid'})

    if syndication:
        G = nx.from_pandas_edgelist(data, 'org_uuid', 'lead_investor_uuids', edge_key = 'distance', edge_attr='investor_count')
    else:
        G = nx.from_pandas_edgelist(data, 'org_uuid', 'lead_investor_uuids')

    P = bipartite.projected_graph(G, data['lead_investor_uuids'])

    # Calculating centrality values for a given metric
    if syndication:
        c_vals = metrics[metric](P, distance='distance')
    else:
        c_vals = metrics[metric](P)
    
    df = pd.DataFrame(c_vals.items(), columns = ['uuid', '{}_centrality'.format(metric)])
    investors = pd.read_csv('../../../data/raw/investors.csv', usecols=['uuid', 'name', 'investor_types'])
    
    merge = pd.merge(df, investors, on='uuid', how='left')
    merge = pd.merge(merge, other_data, on='uuid', how='left')
    # merge.to_csv('vc_centrality.csv', index=False)
    tabloo.show(merge)

    
if __name__=="__main__":
    centrality_t('2020-01-01', delta_t=5, syndication=True)

