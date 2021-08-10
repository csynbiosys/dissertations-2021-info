for i in 1 2 3
do
  for c in closeness betweenness degree eigenvector
  do
     python optunasearch.py "baseline+syn_${c}-${i}" "model_data/baseline+syn/data_${c}.csv" "${i}" "0"
  done
done

python optunasearch.py "baseline-1" "model_data/baseline/data.csv" "1" "0"
python optunasearch.py "baseline-2" "model_data/baseline/data.csv" "2" "0"
