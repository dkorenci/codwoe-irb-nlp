#!/bin/bash
# modified from demo.sh from original GloVe code: https://github.com/stanfordnlp/GloVe.git
# this script is distributed under Apache 2.0 License

set -e
#ARGUMENTS:
# $1 corpus file, with one text, ie token list, per line
# $2 folder with built glove executables
# $3 output file with built glove vectors
# $4 folder for intermediate build resources
CORPUS=$1
echo $1
GLOVE_DIR=$2
echo $2
SAVE_FILE=$3
echo $3
RES_DIR=$4
#  hyperparams
# $5 embedding vector size, $6 ctx. window size, $7 num. train. iterations
VECTOR_SIZE=$5
WINDOW_SIZE=$6 # Number of context words to the left and to the right
MAX_ITER=$7
SEED=$8
X_MAX=10 # cutoff in weighting function
# FIXED SETTINGS
# vocabulary
VOCAB_MIN_COUNT=0 # do not exclude low freq. words, expected input is byte-pair encoding
VOCAB_MAX_COUNT=1000000000
# temporary resource files
VOCAB_FILE=vocab.txt
COOCCURRENCE_FILE=cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
# exec settings
VERBOSE=2
MEMORY=4.0
BINARY=0 # output text file only
NUM_THREADS=4

# EXEC. BUILD PHASES
cd $RES_DIR

echo
echo "$ $GLOVE_DIR/vocab_count -min-count $VOCAB_MIN_COUNT -max-count $VOCAB_MAX_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$GLOVE_DIR/vocab_count -min-count $VOCAB_MIN_COUNT -max-count $VOCAB_MAX_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
echo "$ $GLOVE_DIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
$GLOVE_DIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
echo "$ $GLOVE_DIR/shuffle -memory $MEMORY -verbose $VERBOSE -seed $SEED < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$GLOVE_DIR/shuffle -memory $MEMORY -verbose $VERBOSE -seed $SEED < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
echo "$ $GLOVE_DIR/glove -save-file $SAVE_FILE -seed $SEED -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$GLOVE_DIR/glove -save-file $SAVE_FILE -seed $SEED -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE


#$GLOVE_DIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
#$GLOVE_DIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
#$GLOVE_DIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
#$GLOVE_DIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE

# CLEANUP
rm $VOCAB_FILE
rm $COOCCURRENCE_FILE
rm $COOCCURRENCE_SHUF_FILE