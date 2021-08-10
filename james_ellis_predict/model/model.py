import pandas as pd 
import numpy as np
import xgboost as xgb
import json
import shap
import time
# import tabloo

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

def run_model(wws_centrality=None, syn_centrality=None, n_repeats=3, dependence_plots=False):

    accs = []
    recalls = []
    precisions = []
    df_list = []

    # Logic to point to correct data
    if wws_centrality == None and syn_centrality == None:
        JSON_PATH = 'optimisation_results/baseline'
        DATA_PATH = 'model_data/baseline/data.csv'
        MATCH_STRING = None
    if wws_centrality:
        JSON_PATH = 'optimisation_results/baseline+wws_{}'.format(wws_centrality)
        DATA_PATH = 'model_data/baseline+wws+syn/data_{}.csv'.format(wws_centrality)
        MATCH_STRING = wws_centrality
    if syn_centrality:
        JSON_PATH = 'optimisation_results/baseline+wws_{}'.format(syn_centrality)
        DATA_PATH = 'model_data/baseline+wws+syn/data_{}.csv'.format(syn_centrality)
        MATCH_STRING = syn_centrality
    if wws_centrality and syn_centrality:
        JSON_PATH = 'optimisation_results/baseline+wws_{}+syn_{}'.format(wws_centrality, syn_centrality)
        DATA_PATH = 'model_data/baseline+wws+syn/data_{}_{}.csv'.format(wws_centrality, syn_centrality)
        MATCH_STRING = '{}|{}'.format(wws_centrality, syn_centrality)

    for i in range(n_repeats):
     
        SEED = i + 1
        EXP_JSON_PATH = JSON_PATH + '-{}-hp.json'.format(SEED)

        data = pd.read_csv(DATA_PATH)

        y = data.successful.astype(int)
        X = data.drop(columns=['uuid', 'successful'])
        
        # Split data according to seed provided
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y, random_state=SEED)
    
        # Loading data into DMatrices
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Load optimised parameters
        f = open(EXP_JSON_PATH)
        params = json.load(f)

        num_boost_rounds = params['n_estimators']
        del params['n_estimators']

        # Some extra parameters for all models
        params['objective'] = 'binary:logistic'
        params['scale_pos_weight'] = 3.7
        params['eval_metric'] = 'auc'
        params['booster'] = 'gbtree'
        params['tree_method'] = 'gpu_hist'

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_rounds
        )
    
        y_pred = model.predict(dtest)
        predictions = [round(val) for val in y_pred]

        acc = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        precision = precision_score(y_test, predictions)

        accs.append(acc)
        recalls.append(recall)
        precisions.append(precision)

        if MATCH_STRING is not None:
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            cohorts= {"": shap_values}
            cohort_labels = list(cohorts.keys())
            cohort_exps = list(cohorts.values())

            feature_vals = cohort_exps[0].values
            feature_names = cohort_exps[0].feature_names

            mean_abs_shap = np.average(np.absolute(feature_vals), axis=0)
        
            df = pd.DataFrame({'feature':feature_names, 'mean_abs_shap':mean_abs_shap})
            df = df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
            df['rank'] = df.index.get_level_values(0).values + 1
        
            # Only want centrality feature(s)
            df = df[ df['feature'].str.contains(MATCH_STRING)]
            df_list.append(df)
        
        if dependence_plots:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)

            syn_feature = 'last_round_max_inv_eigenvector_centrality'
            # Dependence of investor centrality along with age_years
            # shap.dependence_plot(syn_feature, shap_values, X_train, interaction_index='age_years', dot_size=4)

            # Dependence of investor centrality along with data completeness
            # num_feats = X_train.shape[1]
            # completeness = X_train.apply(lambda x: x.count(), axis=1)/num_feats
            # completeness = np.where(completeness > 0.93, 1, 0)
        
            # shap.dependence_plot(syn_feature, shap_values, X_train, alpha=0.2, x_jitter=0.004, dot_size=8,other_interaction=completeness)
            
            shap.dependence_plot(syn_feature, shap_values, X_train, dot_size=8, x_jitter=0.004, interaction_index=None)

            
    average_acc = np.average(accs)
    average_recall = np.average(recalls)
    average_precision = np.average(precisions)

    err_acc = np.std(accs)/np.sqrt(n_repeats)
    err_recall = np.std(recalls)/np.sqrt(n_repeats)
    err_precision = np.std(precisions)/np.sqrt(n_repeats)

    print('Results for wws: {}, syn: {}'.format(wws_centrality, syn_centrality))
    print('\tAverage Test Accuracy: {:.4f}+-{:.4f}'.format(average_acc, err_acc))
    print('\tAverage Test Recall: {:.4f}+-{:.4f}'.format(average_recall, err_recall))
    print('\tAverage Test Precision: {:.4f}+-{:.4f}'.format(average_precision, err_precision))

    # i.e. there was a centrality measure in the experiment
    if MATCH_STRING is not None: 

        df_concat = pd.concat(df_list)
        df_concat = df_concat.set_index('feature', drop=False)
        by_row_index = df_concat.groupby(df_concat.index)

        df_concat['mean_shap'] = by_row_index['mean_abs_shap'].mean()
        df_concat['mean_shap_err'] = by_row_index['mean_abs_shap'].std()/np.sqrt(n_repeats)
        df_concat['median_rank'] = by_row_index['rank'].median()

        df_concat = df_concat[['feature','mean_shap', 'mean_shap_err', 'median_rank']].drop_duplicates()
        
        for i in range(df_concat.shape[0]):
            row = df_concat.iloc[i]
        
            print('\tSHAP info on feature: {}'.format(row['feature']))
            print('\t\tAverage of the mean mod shap value: {:.4f}+-{:.4f} '.format(row['mean_shap'], row['mean_shap_err']))
            print('\t\tMedian SHAP rank: {}'.format(row['median_rank']))

if __name__=="__main__":
    # run_model('baseline')

    # centralities = ['closeness', 'betweenness', 'degree', 'eigenvector']
    
   
    # run_model(wws_centrality='closeness', syn_centrality='degree')
   
    run_model(wws_centrality='closeness', syn_centrality='eigenvector', dependence_plots=True)
