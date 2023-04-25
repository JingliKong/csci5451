#!/bin/bash -l

# Fixed parameters for all runs of kmeans
DATADIR="mnist-data"
NCLUST=20
MAXITERS=500
mkdir -p outdirs   # subdir or all output kmeans output

# Full performance benchmark of all combinations of data files and
# processor counts
ALLDATA="digits_all_5e3.txt digits_all_1e4.txt digits_all_3e4.txt"

# # Small sizes for testing
# ALLDATA="digits_all_3e4.txt"

# Iterate over all proc/data file combos
for DATA in $ALLDATA; do
    echo KMEANS $DATA 
    OUTDIR=outdirs/outdir_${DATA}_${NP}
    /usr/bin/time -f "runtime: procs $NP data $DATA realtime %e" \
                  ./kmeans_cuda $DATADIR/$DATA $NCLUST $OUTDIR $MAXITERS
    echo
done

