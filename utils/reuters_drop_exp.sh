OUTPUT_PREFIX=/data22/ivagliano/cit2vec-journal-results/reuters/drop/titles-only
RUN=3
THRESHOLD=20
mkdir -p $OUTPUT_PREFIX
# epochs 20
for RUN in 1 2 3
do
  for DROP in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
  do
    echo python3 ../eval/rcv.py -m $THRESHOLD -dr $DROP -o $OUTPUT_PREFIX/reuters-$THRESHOLD-$RUN-$DROP.txt
    python3 ../eval/rcv.py -m $THRESHOLD -dr $DROP -o $OUTPUT_PREFIX/reuters-$THRESHOLD-$RUN-$DROP.txt
  done
done
exit 0
