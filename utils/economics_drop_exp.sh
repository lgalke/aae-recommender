DATASET_PATH=/data21/lgalke/datasets/econbiz62k.tsv
DATASET_YEAR=2012
OUTPUT_PREFIX=/data22/ivagliano/cit2vec-journal-results/econis/drop/titles-only
THRESHOLD=20
mkdir -p $OUTPUT_PREFIX
for RUN in 1 2 3
do
  for DROP in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
  do
    # With TSV file
    echo python3 ../main.py $DATASET_PATH $DATASET_YEAR -m $THRESHOLD -dr $DROP -o $OUTPUT_PREFIX/econis-$DATASET_YEAR-$THRESHOLD-$RUN-$DROP.txt
    python3 ../main.py $DATASET_PATH $DATASET_YEAR -m $THRESHOLD -dr $DROP -o $OUTPUT_PREFIX/econis-$DATASET_YEAR-$THRESHOLD-$RUN-$DROP.txt
    # With JSON file
    # echo python3 ../eval/econis.py $DATASET_YEAR -m $THRESHOLD -dr $DROP -o $OUTPUT_PREFIX/econis-$DATASET_YEAR-$THRESHOLD-$RUN-$DROP.txt
    # python3 ../eval/econis.py $DATASET_YEAR -m $THRESHOLD -dr $DROP -o $OUTPUT_PREFIX/econis-$DATASET_YEAR-$THRESHOLD-$RUN-$DROP.txt
  done
done
exit 0
