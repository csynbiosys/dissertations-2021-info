Machine learning for reinvest decision prediction: 
Do investors that always re-invest in startups report higher returns than investors that don’t?

I) To setup the environment,you need to install these packages:
    collections
    imblearn
    numpy
    panda
    matplotlib
    sklearn
    tqdm
    seaborn
    xgboost
    
  The data path should follow this:
  |___InvestmentsUK
       |___deals.csv
       |___funding_rounds.csv
       |___funds.csv
       |___jobs.csv
       |___liquidity.csv
       |___organizations.csv
       |___people.csv
       |___shares.csv
  |___1_data_preprocessing.ipynb
  |___2_prediction_models.ipynb
  |___3_prediction_models_optimized.ipynb
  |___4_simulation.ipynb
  |___df.csv
  |___df_tvpi.csv
  |___training_data.csv

II) All the data pre-processing code is included in '1_data_preprocessing.ipynb'.By running this notebook, you could have the 'df.csv', 'df_tvpi.csv' and 'training_data.csv'. Among them, the 'training_data.csv' will be used in the following steps.

III) I use four classification algorithms to construct machine learning models. This part is present in '2_prediction_models.ipynb'. To show how optimization methods work, the comparative experiment is present in '3_prediction_models_optimized.ipynb'

IV) For the simulation investment, you could see three reinvest strategies and the TVPI value they could bring to the seed investors.