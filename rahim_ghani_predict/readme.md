1) Run data-pre-processing-main.ipyNb (INPUT: Crunchbase Original, 2020-QS-World-University-Rankings-compressed.csv, OUTPUT: Preprocessed data)

2) Run dependent-variable-development.ipynb (INPUT:organisations_preprocessed.csv, funding_rounds_preprocessed.csv, OUTPUT: outcome_final_v1.csv)   

3a) Run features-development-people.ipybb (INPUT: outcome_final_v1.csv, jobs_preprocessed.csv, people_preprocessed.csv, degrees_preprocessed.csv, organizations.csv, OUTPUT: features_people.csv) 
3b) Run features-development-ecosystem.ipnyb (INPUT: outcome_final_v1.csv, organisations_preprocessed.csv, Kauffman_Indicators_Early-Stage_Entrepreneurship_Data_2020.csv, OUTPUT: features_ecosystem.csv)   
3c) Run features-development-network.ipynb (INPUT: outcome_final_v1.csv, organisations_preprocessed.csv, event_appearances_preprocessed.csv, OUTPUT: features_network.csv) 
3d) Run features-development-investors.ipynb (INPUT: outcome_final_v1.csv, investors_preprocessed.csv, funding_rounds_preprocessed.csv, OUTPUT: features_investor.csv)   
3e) Run features-finalize.ipynb (INPUT: outcome_final_v1.csv, features_people.csv, features_ecosystem.csv, features_network.csv, features_investors.csv, OUTPUT: features_with_outcome.csv) 

Run use Google Colab for speed
4) Run model_baseline.ipynb (INPUT: features_with_outcome.csv file, OUTPUT: Null) 

All auto-sklearn model run using Google Colab as require Linux Environment
5a) Run model_auto-sklearn_ensemble.ipynb (INPUT: features_with_outcome.csv file, OUTPUT: Null)
5b) Run model_auto-sklearn_gradient-boosting.ipynb (INPUT: features_with_outcome.csv file, OUTPUT: Null)
5c) Run model_auto-sklearn_random-forest.ipynb (INPUT: features_with_outcome.csv file, OUTPUT: Null) 
5d) Run model_auto-sklearn_mlp.ipynb (INPUT: features_with_outcome.csv file, OUTPUT: Null)    

6) Run model_manual_xgboost_final.ipynb (INPUT: features_with_outcome.csv file, OUTPUT: Null)

Link for Google Drive features_with_outcome.csv for Google Colab


