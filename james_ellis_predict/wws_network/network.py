import networkx as nx
import pandas as pd
from networkx.algorithms import bipartite
from networkx.algorithms.centrality import betweenness_centrality, closeness_centrality,\
                                            degree_centrality, eigenvector_centrality

def centrality(df, metric, gpu=False):

    metrics = {
        'closeness': closeness_centrality,
        'betweenness': betweenness_centrality,
        'degree': degree_centrality,
        'eigenvector': eigenvector_centrality
    }

    if gpu:
        import cugraph as cnx
        metrics['betweenness'] = cnx.betweenness_centrality

    G = nx.from_pandas_edgelist(df, 'org_uuid', 'person_uuid')
    P = bipartite.projected_graph(G, df['org_uuid'])

    cent_vals = metrics[metric](P)
    results = pd.DataFrame(cent_vals.items(), columns=['org_uuid', '{}_centrality'.format(metric)])
    
    # CuGraph fails to give some centrality values to a few nodes... fill with 0
    if metric == 'betweenness':
        org = pd.DataFrame(list(P.nodes), columns=['org_uuid'])
        merge = pd.merge(results, org, on='org_uuid', how='right')[['org_uuid', 'betweenness_centrality']].fillna(0)
        return merge

    return results

