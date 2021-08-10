import networkx as nx 
import pandas as pd
from networkx.algorithms import bipartite
from networkx.algorithms.centrality import closeness_centrality, degree_centrality, betweenness_centrality,\
                                            eigenvector_centrality, in_degree_centrality, out_degree_centrality

def syndicate_centrality(df, metric = 'closeness', gpu=False):
    '''
    Creates and measures centrality of the syndicate network
    '''
    metrics = {
            'closeness': closeness_centrality,
            'degree': degree_centrality,
            'betweenness': betweenness_centrality,
            'eigenvector': eigenvector_centrality
        }

    directed_metrics = {
        'indegree': in_degree_centrality,
        'outdegree': out_degree_centrality
    }
    
    if gpu:
        import cugraph as cnx
        metrics['betweenness'] = cnx.betweenness_centrality

    
    # NOTE not using directed metrics in final results due to lack of data
    if metric in [*directed_metrics]:
        
        G = nx.DiGraph()

        # Lead investors in funding rounds with more than one lead investor
        multi_lead = df[ (df['is_lead_investor']==True) & (df['num_lead_investors'] > 1) ]
        G.add_edges_from(zip(multi_lead['investor_uuid'], multi_lead['funding_round_uuid']))
        G.add_edges_from(zip(multi_lead['funding_round_uuid'], multi_lead['investor_uuid']))

        # Adding lead investors
        lead = df[ (df['is_lead_investor'] == True)  & (df['num_lead_investors'] == 1)]
        G.add_edges_from(zip(lead['investor_uuid'], lead['funding_round_uuid']))

        # Adding non-lead investors in funding rounds with lead investors present
        non_lead = df[ (df['is_lead_investor'] == False) & (df['num_lead_investors'] >=1) ]
        G.add_edges_from(zip(non_lead['funding_round_uuid'], non_lead['investor_uuid']))

        # NOTE this assumption could be total dogshit
        # Adding non-lead investors in funding rounds with no lead investors assigned
        non_lead_nolead = df[ (df['is_lead_investor'] == False) & (df['num_lead_investors'] == 0) ]
        G.add_edges_from(zip(non_lead_nolead['funding_round_uuid'], non_lead_nolead['investor_uuid']))
        G.add_edges_from(zip(non_lead_nolead['investor_uuid'], non_lead_nolead['funding_round_uuid']))

        # Project network onto the investor nodes
        P = bipartite.projected_graph(G, df['investor_uuid'].unique())

        cent_vals = directed_metrics[metric](P)
        results = pd.DataFrame(cent_vals.items(), columns=['investor_uuid', '{}_centrality'.format(metric)])

        return results

    else:
        G = nx.Graph()
        G = nx.from_pandas_edgelist(df, 'investor_uuid', 'funding_round_uuid')

        # Projection onto investor nodes
        P = bipartite.projected_graph(G, df['investor_uuid'].unique())

        cent_vals = metrics[metric](P)
        results = pd.DataFrame(cent_vals.items(), columns=['investor_uuid', '{}_centrality'.format(metric)])

        # CuGraph fails to give some centrality values to a few nodes... fill with 0
        if metric == 'betweenness':
            investors = pd.DataFrame(list(P.nodes), columns=['investor_uuid'])
            merge = pd.merge(results, investors, on='investor_uuid', how='right')[['investor_uuid', 'betweenness_centrality']].fillna(0)
            return merge

        return results

if __name__=="__main__":
    centrality(metric='outdegree')

