DATASET_PATH=../Data/Economics/econbiz62k.tsv
DATASET_YEAR=2012
OUTPUT_PREFIX=final_results/run2/economics
mkdir -p $OUTPUT_PREFIX
for THRESHOLD in 1 2 3 4 5 10 15 20 25
do
  python3 main.py $DATASET_PATH $DATASET_YEAR -m $THRESHOLD -o $OUTPUT_PREFIX/m$THRESHOLD.log
done
exit 0
