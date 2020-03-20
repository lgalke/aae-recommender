OUTPUT_PREFIX=/data22/ivagliano/cit2vec-journal-results/mpd/vae-dae/generic-condition
RUN=3
mkdir -p $OUTPUT_PREFIX

# epochs 55
for THRESHOLD in 55 50 45 40 35 30
do
  echo python3 ../eval/mpd/mpd.py -m $THRESHOLD -o $OUTPUT_PREFIX/mpd-$THRESHOLD-$RUN.txt
  python3 ../eval/mpd/mpd.py -m $THRESHOLD -o $OUTPUT_PREFIX/mpd-$THRESHOLD-$RUN.txt
done
exit 0
