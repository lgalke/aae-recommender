# created a new script to avoid starange behaviour as the first dblp run is still ongoing (see link below)
# https://stackoverflow.com/questions/3398258/edit-shell-script-while-its-running
DATASET_YEAR=2014
OUTPUT_PREFIX=/data22/ivagliano/cit2vec-journal-results
DATASET=acm
RUN=2
mkdir -p $OUTPUT_PREFIX
# epochs 20 for acm as for dblp
for THRESHOLD in 55 50 45 40 35 30 25 20 15 10
do
  echo python3 aminer.py $DATASET_YEAR -d $DATASET -m $THRESHOLD -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRESHOLD-$RUN.txt
  python3 aminer.py $DATASET_YEAR -d $DATASET -m $THRESHOLD -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRESHOLD-$RUN.txt
done
exit 0
