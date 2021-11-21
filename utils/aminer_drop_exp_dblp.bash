#!/usr/bin/env bash

# Symlink to aminer and vectors exist
# ../aminer -> /data22/ivagliano/aminer
# ../vectors/GoogleNews-vectors-negative300.bin.gz -> /data21/lgalke/vectors/GoogleNews-vectors-negative300.bin.gz

# STOP ON ERROR
set -e 

AMINER_PY="eval/aminer.py"

echo "Branch: $(git branch | grep '^*')"

echo "Checking aminer symlink to data:"
ls -l1 . | grep "aminer"

echo "Checking symlink to word vectors:"
ls -l1 ./vectors/ | grep "GoogleNews"


DATASET="dblp"
YEAR=2017 # DONE: Verify that 2018 is correct split year for DBLP -> it was 2017.....
MINCOUNT=55 # TODO ask Iacopo
# RESULTS_DIR="results-drop-$DATASET-$YEAR-m$MINCOUNT-metadata"
RESULTS_DIR="results-drop-$DATASET-$YEAR-m$MINCOUNT-title-2"

echo "Using dataset $DATASET with split on year $YEAR and min count $MINCOUNT"

echo "Starting experiments..."

for RUN in "2" "3"; do
	RESULTS_DIR="results-drop-$DATASET-$YEAR-m$MINCOUNT-title-$RUN"
	echo "Using dir '$RESULTS_DIR' to store results"
	mkdir -p "$RESULTS_DIR"
	for DROP in "0.6" "0.7" "0.8" "0.9"; do
		OUTPUT_FILE="$RESULTS_DIR"/"$DATASET-$YEAR-m$MINCOUNT-drop$DROP.txt"
		echo "python3 $AMINER_PY $YEAR --drop $DROP -d $DATASET -m $MINCOUNT -o $OUTPUT_FILE --baselines --autoencoders --conditioned_autoencoders"
		python3 "$AMINER_PY" "$YEAR" --drop "$DROP" -d "$DATASET" -m "$MINCOUNT" -o "$OUTPUT_FILE" --baselines --autoencoders --conditioned_autoencoders
	done

	RESULTS_DIR="results-drop-$DATASET-$YEAR-m$MINCOUNT-metadata-$RUN"
	echo "Using dir '$RESULTS_DIR' to store results"
	mkdir -p "$RESULTS_DIR"
	for DROP in "0.6" "0.7" "0.8" "0.9"; do
		OUTPUT_FILE="$RESULTS_DIR"/"$DATASET-$YEAR-m$MINCOUNT-drop$DROP.txt"
		echo "python3 $AMINER_PY $YEAR --drop $DROP -d $DATASET -m $MINCOUNT -o $OUTPUT_FILE --all_metadata --conditioned_autoencoders"
		python3 "$AMINER_PY" "$YEAR" --drop "$DROP" -d "$DATASET" -m "$MINCOUNT" -o "$OUTPUT_FILE" --all_metadata --conditioned_autoencoders
	done
done


