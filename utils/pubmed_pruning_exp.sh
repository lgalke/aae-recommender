DATASET_PATH=../Data/PMC/citations_pmc.tsv
DATASET_YEAR=2011
OUTPUT_PREFIX=final_results/run3/pubmed
mkdir -p $OUTPUT_PREFIX
for THRESHOLD in 55 50 45 40 35 30 25 20 15
do
  python3 main.py $DATASET_PATH $DATASET_YEAR -m $THRESHOLD -o $OUTPUT_PREFIX/m$THRESHOLD.log
done
exit 0
