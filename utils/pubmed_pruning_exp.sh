DATASET_PATH=/data22/ggerstenkorn/citation_data_preprocessing/final_data/owner_list_cleaned.csv
DATASET_YEAR=2011
OUTPUT_PREFIX=/data22/ivagliano/cit2vec-journal-results/pubmed/generic-condition
RUN=3
mkdir -p $OUTPUT_PREFIX
for THRESHOLD in 50 45 40 35 30 25 20 15 10
do
  echo python3 ../main.py $DATASET_PATH $DATASET_YEAR -m $THRESHOLD -o $OUTPUT_PREFIX/pubmed-$DATASET_YEAR-$THRESHOLD-$RUN-all-title-pretrained.txt
  python3 ../main.py $DATASET_PATH $DATASET_YEAR -m $THRESHOLD -o $OUTPUT_PREFIX/pubmed-$DATASET_YEAR-$THRESHOLD-$RUN-all-title-pretrained.txt
done
exit 0
