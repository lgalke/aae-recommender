#DATASET_PATH=/data22/ggerstenkorn/citation_data_preprocessing/final_data/owner_list_cleaned.csv
DATASET_PATH="/media/nvme1n1/lgalke/datasets/AAEREC/pmc_final_data/owner_list_cleaned-SHORT.csv"
DATASET_YEAR=2011
OUTPUT_PREFIX="results/pubmed_short_NO-MESH"
# RUN=1
THRESHOLD=55
DROP="0.8"
mkdir -p $OUTPUT_PREFIX

set -e 

for RUN  in 1 2 3
do
    echo "python3 main_pubmed_nomesh.py $DATASET_PATH $DATASET_YEAR -m $THRESHOLD -dr $DROP -o $OUTPUT_PREFIX/pubmed-short-nomesh-$DATASET_YEAR-$THRESHOLD-$RUN-$DROP.txt"
    python3 main_pubmed_nomesh.py $DATASET_PATH $DATASET_YEAR -m $THRESHOLD -dr $DROP -o $OUTPUT_PREFIX/pubmed-short-nomesh-$DATASET_YEAR-$THRESHOLD-$RUN-$DROP.txt
done
exit 0
