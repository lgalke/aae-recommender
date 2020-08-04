#!/usr/bin/env bash

### PubMed
echo "PubMed"
for pruning in {15..55..5}; do
    python3 compute_pairwise_mi.py "/data21/lgalke/datasets/citations_pmc.tsv" --min-count "$pruning"
done

#### DBLP
echo "DBLP"
for pruning in {20..55..5}; do
    python3 eval/aminer.py 9999 --compute-mi --dataset "dblp" --min-count "$pruning"
done

### ACM
echo "ACM"
for pruning in {15..55..5}; do
    python3 eval/aminer.py 9999 --compute-mi --dataset "acm" --min-count "$pruning"
done

### EconBiz
echo "EconBiz"
for pruning in {1..5} {10..20..5}; do
    python3 eval/econis.py 9999 --compute-mi --min-count "$pruning"
done

### IREON
echo "IREON"
for pruning in {1..5} {10..20..5}; do
    python3 eval/fiv.py 9999 --compute-mi --min-count "$pruning"
done

### Reuters
echo "Reuters"
for pruning in {1..5} {10..20..5}; do
    python3 eval/rcv.py --compute-mi --min-count "$pruning"
done
