for i in 1 2 3
do
  for c in closeness betweenness degree eigenvector
  do
     python optunasearch.py "baseline+wws_${c}-${i}" "model_data/baseline+wws/data_${c}.csv" "${i}" "1"
  done
done

python optunasearch.py "baseline-3" "model_data/baseline/data.csv" "3" "1"

