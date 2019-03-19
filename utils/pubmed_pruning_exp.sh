DATASET_PATH=/data21/lgalke/datasets/citations_pmc.tsv
DATASET_YEAR=2011
OUTPUT_PREFIX=/data22/ivagliano/cit2vec-journal-results/pubmed-vae-dae/titles-only
RUN=1
mkdir -p $OUTPUT_PREFIX
for THRESHOLD in 50 45 40 35 30 25 20 15 10
do
  echo python3 ../main.py $DATASET_PATH $DATASET_YEAR -m $THRESHOLD -o $OUTPUT_PREFIX/pubmed-$DATASET_YEAR-$THRESHOLD-$RUN.txt
  python3 ../main.py $DATASET_PATH $DATASET_YEAR -m $THRESHOLD -o $OUTPUT_PREFIX/pubmed-$DATASET_YEAR-$THRESHOLD-$RUN.txt
done
exit 0
