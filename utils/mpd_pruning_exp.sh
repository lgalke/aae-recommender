OUTPUT_PREFIX=/data22/ivagliano/cit2vec-journal-results/mpd/aggregate-1-cond
RUN=1
mkdir -p $OUTPUT_PREFIX
# epochs 55
for THRESHOLD in 45 40 35 30 25
do
  echo python3 ../eval/mpd/mpd.py -a -m $THRESHOLD -o $OUTPUT_PREFIX/mpd-$THRESHOLD-$RUN.txt
  python3 ../eval/mpd/mpd.py -a -m $THRESHOLD -o $OUTPUT_PREFIX/mpd-$THRESHOLD-$RUN.txt
done
exit 0
