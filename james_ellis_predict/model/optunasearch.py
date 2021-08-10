import optuna 
import xgboost as xgb
import pandas as pd
import argparse
# import tabloo
import joblib

from sklearn.model_selection import train_test_split

class Objective(object):
    def __init__(self, data_path, gpu_id, data_split_seed=1):
        self.data_path = data_path
        self.gpu_id = gpu_id
        self.data_split_seed = data_split_seed
        
    def __call__(self, trial):
        print(self.gpu_id)
        df = pd.read_csv(self.data_path)

        y = df['successful']
        X = df.drop(columns=['successful', 'uuid'])

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y, random_state=int(self.data_split_seed))
        
        # Converting from pandas to dmatrix for xgboost
        dtrain = xgb.DMatrix(X_train, label=y_train)

        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "scale_pos_weight": 3.7,
            "booster": "gbtree",
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.2, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.01, 20.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 9),
            "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        }

        if self.gpu_id != -1:
            param["tree_method"] = "gpu_hist"
            param["gpu_id"] = int(self.gpu_id)
        
        else:
            param["tree_method"] = "hist"
        
        cv_results = xgb.cv(
            params=param,
            dtrain=dtrain,
            num_boost_round=10000,
            nfold=3,
            stratified=True,
            early_stopping_rounds=50,
        )

        # Set n_estimators as a trial attribute; Accessible via study.trials_dataframe().
        trial.set_user_attr("n_estimators", len(cv_results))

        best_score = cv_results["test-auc-mean"].values[-1]
        return best_score

if __name__=="__main__":
    
    def none_or_str(value):
        if value == 'None':
            return None 
        return value

    parser = argparse.ArgumentParser()
    # Optional arguments in command line
    parser.add_argument('studyname', nargs='?')
    parser.add_argument('datapath', nargs='?')
    parser.add_argument('data_split_seed', nargs='?')
    parser.add_argument('gpu_id', nargs='?')
    args = parser.parse_args()

    STUDYNAME = args.studyname
    DATAPATH = args.datapath
    SPLITSEED = args.data_split_seed
    GPU_ID = args.gpu_id

    study = optuna.create_study(study_name=STUDYNAME, direction="maximize")
   
    study.optimize(Objective(DATAPATH, GPU_ID, data_split_seed=SPLITSEED), timeout=3600)

    # Save results
    results = study.trials_dataframe()
    results.to_csv('optimisation_results/{}.csv'.format(STUDYNAME), index=False)

    # Save study incase needs resuming
    joblib.dump(study, "optimisation_results/{}.pkl".format(STUDYNAME))
