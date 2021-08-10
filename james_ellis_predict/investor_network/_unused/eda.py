import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr 

df = pd.read_csv('vc_centrality.csv')

groups = ['angel', 'pre_seed', 'seed', 'series_a', 'series_b', 'series_c', 'series_d', 'series_e']


for g in groups:
    data = df[df['most_common_invest_type']==g]['closeness_centrality']
    print('Investor Type: {}, Average Centrality: {:.2f}, IQR: {:.2f}'.format(g, np.average(data), iqr(data)))

