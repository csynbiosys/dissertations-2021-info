import pandas as pd 
import numpy as np
from itertools import product
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

def splitDataFrameList(df,target_column,separator):
    ''' df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 
    The values in the other columns are duplicated across the newly divided rows.
    '''
    def splitListToRows(row,row_accumulator,target_column,separator):
        split_row = row[target_column].split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))
    new_df = pd.DataFrame(new_rows)
    return new_df


def dict_product(d, random_seed=42):
    keys = d.keys()
    dict_list = []
    for element in product(*d.values()):
        dict_list.append(dict(zip(keys, element)))
    
    np.random.seed(random_seed)
    np.random.shuffle(dict_list)
    
    return dict_list

def xgb_multi(y, t, threshold=0.5):
    t = t.get_label()
    y_bin = (y > threshold).astype(int)
    # Last metric in the list is used for early stopping
    return [('accuracy', accuracy_score(t, y_bin)), ('recall', recall_score(t, y_bin)), 
            ('precision', precision_score(t, y_bin)), ('f1',f1_score(t,y_bin))]

def xgb_multi_weighted(y, t, threshold=0.5):

    def _replaceitem(x):
        if x>0:
            return 3.7
        return 1

    t = t.get_label()
    y_bin = (y > threshold).astype(int)
    weight = list(map(_replaceitem, t))

    # Last metric in the list is used for early stopping
    return [('accuracy', accuracy_score(t, y_bin)), ('recall', recall_score(t, y_bin)), 
            ('precision', precision_score(t, y_bin)), ('f1',f1_score(t,y_bin, sample_weight=weight))]