DATASET_YEAR=2016
OUTPUT_PREFIX=/data22/ivagliano/cit2vec-journal-results/ireon/drop/titles-only
# RUN=1
THRESHOLD=20
mkdir -p $OUTPUT_PREFIX
for RUN  in 1 2 3
do
  for DROP in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 #0.9
  do
    echo python3 ../eval/fiv.py $DATASET_YEAR -m $THRESHOLD -dr $DROP -o $OUTPUT_PREFIX/ireon-$DATASET_YEAR-$THRESHOLD-$RUN-$DROP.txt
    #python3 ../eval/fiv.py $DATASET_YEAR -m $THRESHOLD -dr $DROP -o $OUTPUT_PREFIX/ireon-$DATASET_YEAR-$THRESHOLD-$RUN-$DROP.txt
  done
done
exit 0
