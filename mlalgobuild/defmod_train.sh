#     MAIN_COMMAND="${COMMAND} ${TASK} ${ACTION_F} --model \"${MODEL_BASE_NAME}\" --settings \"${MODEL_SETTINGS}\" --train_file \"${DATA_TRAIN}\" --dev_file \"${DATA_DEV}\" --test_file \"${DATA_TEST}\" --device \"cuda:${CUDA}\" --spm_model_path \"${SPM_PATH}\" --save_dir \"${SAVE_DIR}\" --pred_file \"${PRED_FILE}\" --summary_logdir \"${LOG_DIR}\" ${MODEL_PATH_F} ${EMBED_F} ${INPUT_KEY_F} ${OUTPUT_KEY_F} ${EXPERIMENT_ID_F} ${PARAMETERS_F} ${EPOCHS_F} ${HEADS_F} ${LAYERS_F}"
REPO_ROOT=`readlink -f ..`
export PYTHONPATH="$REPO_ROOT:$PYTHONOATH"
echo $PYTHONPATH

#      --train_file  "/datafast/codwoe/train-data_all/en.train.json" \
#      --dev_file "/datafast/codwoe/train-data_all/en.dev.json" \
# --word_emb "glove"
# "defmod-rnn"
# --vocab_subdir "dset_v1"

#rm -rf logs
#rm -rf models

# SETUP INPUT AND DICT DATA
TRAINDEVDSET="dset_v1"
trainfile="/datafast/codwoe/dataset/$TRAINDEVDSET/en.train.json"
devfile="/datafast/codwoe/dataset/$TRAINDEVDSET/en.dev.json"

# DEFINE CORPUS FOR DICTIONARY TRAINING (if smp is used) - CAN BE SEPARATE FROM INPUT CORPUS
# SAME CORPUS IS USED FOR GLOVE TRAINING (if glove is used)
vocabdset=$TRAINDEVDSET

PRED_FILE="/datafast/codwoe/trial-data_all/en.trial.complete.json"

rm -rf logs
rm -rf models

python main.py \
      --do_train --save_dir "models" --summary_logdir "logs" \
      --train_file "$trainfile" --dev_file "$devfile" \
      --vocab_lang "en" --vocab_subdir "$vocabdset" --vocab_size 8000 \
      --device "cuda" --epochs 3 --dropout 0.3 --in_dropout 0.1 --learn_rate 0.0005 --scheudle 'plat' \
      --model "defmod-rnn" --rnn_arch 'gru' --input_key "allvec" --allvec_mode "concat" --output_key "gloss" \
      --settings "defmod-base-xen" --n_head 2 --n_layers=2 \
      --word_emb "glove" --emb_size 256 --maxlen 10 \
      --do_pred --model_path "models/best/model.pt" --pred_file "$PRED_FILE" --pred_output "en.trial.pred.json" \
      --modout "dsetv1" --pred_out_align True \
