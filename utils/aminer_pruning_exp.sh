DATASET_YEAR=2017
OUTPUT_PREFIX=/data22/ivagliano/cit2vec-journal-results
DATASET=dblp
RUN=1
mkdir -p $OUTPUT_PREFIX
# for dblp 55 done run by hannd
# epochs should be always 20 for dblp, 100 for others, (we should verify runtime with acm)
for THRESHOLD in 25 20 15 10
do
  echo python3 ../eval/aminer.py $DATASET_YEAR -d $DATASET -m $THRESHOLD -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRESHOLD-$RUN.txt
  python3 aminer.py $DATASET_YEAR -d $DATASET -m $THRESHOLD -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRESHOLD-$RUN.txt
done
exit 0
