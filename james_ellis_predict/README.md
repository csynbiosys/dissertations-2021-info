# james_ellis_predict



## Required Modules

Using the conda package manager.

```bash
conda install pandas
conda install -c conda-forge matplotlib
conda install -c conda-forge scipy
conda install -c conda-forge networkx
conda install -c conda-forge scikit-learn
conda install -c conda-forge optuna
conda install -c conda-forge xgboost
conda install -c conda-forge shap
```
if you have CUDA enabled device
```bash
conda install -c rapidsai cugraph
```

## Main Folders
investor_network: contains the code to measure the centrality of investors in a syndicate network. I output to .csv files due to the time for processing some centralities (closeness takes nearly 3 hours). 

wws_network: contains the code to measure the centrality of startups following Bonaventura's method. Again, saved to .csv files.

model: The .csv files are imported into the folder model/network_centralities. This folder contains all the code necessary to train and test the XGBOOST classifier. 

NOTE: any subfolders marked _unused can be ignored.

## Data Imports from Dropbox
- model_data: place in model/ folder
- network_centralities: place in model/ folder

NOTE: assuming we are in the directory where my 3 main folders are, the crunchbase data has to be in ../../data/raw

## investor_network
Run main.py with the start of the simulation window in the format yyyy-mm-dd (string) provided. Please also indicate the presence of GPU and a centrality measure.

Example in main.py

```python
if __name__=="__main__":
    run('2017-12-15', metric='betweenness', gpu=False)
```
.csv files will save to the investor_network folder

## wws_network
Similarly, run main.py with the start of the simulation window in the format yyyy-mm-dd (string) provided.  Please also indicate the presence of GPU and a centrality measure.

```python
if __name__=="__main__":
    wws_centralities('2017-12-15', metric='betweenness', gpu=False)
```
## Supported Centralities

Choose one of closeness, betweenness, degree, or eigenvector for either network.
GPU support only for betweenness centrality.

## Model
I run several models including
- Baseline
- Baseline + Syndicate Network
- Baseline + WWS Network (Bonaventura's network)
- Baseline + Syndicate + WWS

The baseline features are created following Arroyo et al. and are split into three python files: company_info, funding_info, and founders_info. Another file called target_variable.py calculates the dependent variable i.e. successful or not. We follow Arroyo's definition of success. 

model_data.py stitches together all independent and dependent variables together, including any network centralities. 

For example, in model_data.py, to obtain data for every model we would run

```python
if __name__=="__main__":
    # Dates for start of warmup window, start of simulation window, and end of simulation window respectively.
    dates = ['2013-12-15','2017-12-15', '2020-12-15']
    cs = ['betweenness', 'closeness', 'degree', 'eigenvector']

    # Baseline
    model_data(*dates, save_path='model_data/baseline/data.csv')

    # Baseline + syndicate network
    for c in cs:
         model_data(*dates, syn_centrality=c, save_path='model_data/baseline+syn/data_{}.csv'.format(c))

    # Baseline + wws network
    for c in cs:
        model_data(*dates, wws_centrality=c, save_path='model_data/baseline+wws/data_{}.csv'.format(c))
    
    # Baseline + wws closeness + syn degree
    model_data(*dates, wws_centrality='closeness', syn_centrality='degree', save_path='model_data/baseline+wws+syn/data_closeness_degree.csv')
```

Once these files are saved to a suitable location, an extensive hyperparameter search is performed for each model. I provide an example in optunasearch.py but would not recommend reproducing the full results because it takes around 30 hours. Although, in line 85 of optunasearch.py:
```python
study.optimize(Objective(DATAPATH, GPU_ID, data_split_seed=SPLITSEED), timeout=3600)
```
changing ```timeout``` to a much shorter time will speed up the process.  

For example,
```bash
# python optunasearch.py "EXPERIMENT_NAME" "DATA_LOCATION" "RANDOM_SEED" "GPU_ID"
python optunasearch.py "baseline-1" "model_data/baseline/data.csv" "1" "0"
```
If you have no GPU support, change GPU_ID "0" to "-1" and it will change to CPU.

Hyperparameter results can be found in optimisation_results, along with the set of parameters that maximised a 3 fold stratified cross-validation AUC.

Final model results (accuracy, recall, precision, SHAP values), are all obtained using model.py, for example:

```python
if __name__=="__main__":
    # Baseline
    run_model()

    # Baseline + syndicate + WWS
    run_model(wws_centrality='closeness', syn_centrality='degree')
```
