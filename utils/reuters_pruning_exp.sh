OUTPUT_PREFIX=/data22/ivagliano/cit2vec-journal-results/reuters-vae-dae/titles-only
RUN=3
mkdir -p $OUTPUT_PREFIX
# epochs 20
for THRESHOLD in 20 15 10 5 4 3 2 1
do
  echo python3 ../eval/rcv.py -m $THRESHOLD -o $OUTPUT_PREFIX/reuters-$THRESHOLD-$RUN.txt
  python3 ../eval/rcv.py -m $THRESHOLD -o $OUTPUT_PREFIX/reuters-$THRESHOLD-$RUN.txt
done
exit 0
