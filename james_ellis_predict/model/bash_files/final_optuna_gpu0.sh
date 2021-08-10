for i in 1 2 3
do
  python optunasearch.py "baseline+wws_closeness+syn_degree-${i}" "model_data/baseline+wws+syn/data_closeness_degree.csv" "${i}" "0"
done
