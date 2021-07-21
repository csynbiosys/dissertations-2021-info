import networkx as nx
import pandas as pd
from job_transfer import graph_data
from networkx.algorithms import bipartite
from networkx.algorithms.centrality import closeness_centrality
import time

dates = [_ for _ in range(2000,2010)]

for date in dates:

    t = time.time()
    data = graph_data(date)
    G = nx.from_pandas_edgelist(data, 'org_uuid', 'person_uuid')
    P = bipartite.projected_graph(G, data['org_uuid'])

    bc = closeness_centrality(P)
    df = pd.DataFrame(bc.items(), columns=['org_uuid', 'betweenness_centrality'])
    df.to_csv('data/results/centrality_values/cc_{}.csv'.format(date), index=False)

    print(time.time()-t)
    
