import glob
import pandas as pd 
import tabloo
import json

def remove_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text

def remove_suffix(text, suffix):
    return text[:-len(suffix)] if text.endswith(suffix) else text

def get_optimal_hp(csv_file):
    df = pd.read_csv(csv_file)
    df = df.sort_values('value', ascending=False).head(1)

    params = [i for i in df.columns.values if i.startswith('params')]

    # Build dictionary to save to json
    param_dict = {}
    param_dict['n_estimators'] = df['user_attrs_n_estimators'].tolist()[0]

    for param in params:
        param_dict[remove_prefix(param, 'params_')] = df[param].tolist()[0]

    return param_dict

if __name__=="__main__":
    
    files = glob.glob('*.csv')
    
    for f in files:
        
        save_name = remove_suffix(f, '.csv') + '-hp.json'
    
        json_dict = get_optimal_hp(f)
       
        with open(save_name, 'w') as fp:
            json.dump(json_dict, fp)    