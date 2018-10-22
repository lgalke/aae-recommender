DATASET_YEAR=2016
OUTPUT_PREFIX=/data22/ivagliano/cit2vec-journal-results
RUN=1
mkdir -p $OUTPUT_PREFIX
# epochs 20
for THRESHOLD in 50 45 40 35 30 25 20 15 10
do
  echo python3 ../eval/fiv.py $DATASET_YEAR -m $THRESHOLD -o $OUTPUT_PREFIX/ireon-$DATASET_YEAR-$THRESHOLD-$RUN.txt
  python3 ../eval/fiv.py $DATASET_YEAR -m $THRESHOLD -o $OUTPUT_PREFIX/ireon-$DATASET_YEAR-$THRESHOLD-$RUN.txt
done
exit 0
