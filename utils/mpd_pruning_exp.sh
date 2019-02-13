OUTPUT_PREFIX=/data22/ivagliano/cit2vec-journal-results/mpd/titles-only
RUN=2
mkdir -p $OUTPUT_PREFIX
# epochs 55
for THRESHOLD in 50 45 40 35 30
do
  echo python3 ../eval/mpd/mpd.py -m $THRESHOLD -o $OUTPUT_PREFIX/mpd-$THRESHOLD-$RUN.txt
  python3 ../eval/mpd/mpd.py -m $THRESHOLD -o $OUTPUT_PREFIX/mpd-$THRESHOLD-$RUN.txt
done
exit 0
