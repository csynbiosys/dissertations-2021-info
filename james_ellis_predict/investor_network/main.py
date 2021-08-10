import time 
from syndicates import syndicates
from network import syndicate_centrality

def run(ts, metric='closeness', gpu=False):
    data = syndicates(ts, relation_length=5)
    results = syndicate_centrality(data, metric=metric, gpu=gpu)

    results.to_csv('syndicate_{}_centralities.csv'.format(metric))

    return results

if __name__=="__main__":

    # metrics = ['degree', 'betweenness', 'eigenvector', 'closeness']
    # for m in metrics:
    #     t1 = time.time()
    #     r = run('2017-12-15', metric=m)
    #     print('Metric: {}'.format(m))
    #     print('\tSize: {}, Num unique investors: {}'.format(r.shape[0], r['investor_uuid'].nunique()))
    #     print('\tTime for calculation: {}'.format(time.time() - t1))

    run('2010-12-15', metric='betweenness', gpu=False)