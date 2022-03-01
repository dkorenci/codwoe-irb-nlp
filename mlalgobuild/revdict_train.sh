#     MAIN_COMMAND="${COMMAND} ${TASK} ${ACTION_F} --model \"${MODEL_BASE_NAME}\" --settings \"${MODEL_SETTINGS}\" --train_file \"${DATA_TRAIN}\" --dev_file \"${DATA_DEV}\" --test_file \"${DATA_TEST}\" --device \"cuda:${CUDA}\" --spm_model_path \"${SPM_PATH}\" --save_dir \"${SAVE_DIR}\" --pred_file \"${PRED_FILE}\" --summary_logdir \"${LOG_DIR}\" ${MODEL_PATH_F} ${EMBED_F} ${INPUT_KEY_F} ${OUTPUT_KEY_F} ${EXPERIMENT_ID_F} ${PARAMETERS_F} ${EPOCHS_F} ${HEADS_F} ${LAYERS_F}"
REPO_ROOT=`readlink -f ..`
export PYTHONPATH="$REPO_ROOT:$PYTHONOATH"
echo $PYTHONPATH

TRAINDEVDSET="orig_lc"
trainfile="/datafast/codwoe/dataset/$TRAINDEVDSET/en.train.json"
devfile="/datafast/codwoe/dataset/$TRAINDEVDSET/en.dev.json"
vocabdset=$TRAINDEVDSET

# file for which to run predictions
PRED_FILE="/datafast/codwoe/trial-data_all/en.trial.complete.json"

python main.py \
      --do_train  --save_dir "models" --summary_logdir "logs" \
      --train_file "$trainfile" --dev_file "$devfile" \
      --vocab_lang "en" --vocab_subdir "$vocabdset" --vocab_size 8000 \
      --device "cuda" --epochs 2 \
      --model "revdict-base" --input_key "gloss" --output_key "electra" \
      --settings "revdict-base-mse" --n_head 2 --n_layers=2 --emb_size 128 \
      --word_emb "glove" \
      --do_pred --pred_file "$PRED_FILE" --pred_output "en.trial.pred.json" --pred_mod "lower" \
      --model_path "models/best/model.pt"
