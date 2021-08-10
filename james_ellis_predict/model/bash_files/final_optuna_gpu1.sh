for i in 1 2 3
do
  python optunasearch.py "baseline+wws_closeness+syn_eigenvector-${i}" "model_data/baseline+wws+syn/data_closeness_eigenvector.csv" "${i}" "1"
done
