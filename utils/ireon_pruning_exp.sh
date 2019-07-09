DATASET_YEAR=2016
OUTPUT_PREFIX=/data22/ivagliano/cit2vec-journal-results/ireon/generic-condition
RUN=3
mkdir -p $OUTPUT_PREFIX
# epochs 20
for THRESHOLD in 20 15 10 5 4 3 2 1
do
  echo python3 ../eval/fiv.py $DATASET_YEAR -m $THRESHOLD -o $OUTPUT_PREFIX/ireon-$DATASET_YEAR-$THRESHOLD-$RUN.txt
  python3 ../eval/fiv.py $DATASET_YEAR -m $THRESHOLD -o $OUTPUT_PREFIX/ireon-$DATASET_YEAR-$THRESHOLD-$RUN.txt
done
exit 0
