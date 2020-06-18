#!/usr/bin/env bash

## PubMed
echo "PubMed"
for pruning in {15..55..5}; do
    python3 stats.py --dataset "pubmed" --min-count "$pruning"
done

#### DBLP
echo "DBLP"
for pruning in {20..55..5}; do
    python3 stats.py --dataset "dblp" --min-count "$pruning"
done

### ACM
echo "ACM"
for pruning in {15..55..5}; do
    python3 stats.py --dataset "acm" --min-count "$pruning"
done

### EconBiz
echo "EconBiz"
for pruning in {1..5} {10..20..5}; do
    python3 stats.py --dataset "econbiz" --min-count "$pruning"
done

### IREON
echo "IREON"
for pruning in {1..5} {10..20..5}; do
    python3 stats.py --dataset "swp" --min-count "$pruning"
done

### Reuters
echo "Reuters"
for pruning in {1..5} {10..20..5}; do
    python3 stats.py --dataset "rcv" --min-count "$pruning"
done

