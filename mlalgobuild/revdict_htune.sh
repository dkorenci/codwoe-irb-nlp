#     MAIN_COMMAND="${COMMAND} ${TASK} ${ACTION_F} --model \"${MODEL_BASE_NAME}\" --settings \"${MODEL_SETTINGS}\" --train_file \"${DATA_TRAIN}\" --dev_file \"${DATA_DEV}\" --test_file \"${DATA_TEST}\" --device \"cuda:${CUDA}\" --spm_model_path \"${SPM_PATH}\" --save_dir \"${SAVE_DIR}\" --pred_file \"${PRED_FILE}\" --summary_logdir \"${LOG_DIR}\" ${MODEL_PATH_F} ${EMBED_F} ${INPUT_KEY_F} ${OUTPUT_KEY_F} ${EXPERIMENT_ID_F} ${PARAMETERS_F} ${EPOCHS_F} ${HEADS_F} ${LAYERS_F}"
REPO_ROOT=`readlink -f ..`
export PYTHONPATH="$REPO_ROOT:$PYTHONOATH"
echo $PYTHONPATH

# ROOT="/datafast/codwoe"
ROOT="/root/codwoe-irb-nlp"
OUTPUT="$ROOT/output/revdict"
DATASET="$ROOT/resources/dataset"
TRAINDEVDSET="orig_lc"
EMBBEDING="electra"
LANGUAGE="en"
trainfile="$DATASET/$TRAINDEVDSET/$LANGUAGE.train.json"
devfile="$DATASET/$TRAINDEVDSET/$LANGUAGE.dev.json"
vocabdset=$TRAINDEVDSET

models="$OUTPUT/models/$EMBBEDING/$LANGUAGE"
logs="$OUTPUT/logs/$EMBBEDING/$LANGUAGE"

# file for which to run predictions
pred_file="$DATASET/trial-data_all/$LANGUAGE.trial.complete.json"
pred_output="$OUTPUT/prediction/$LANGUAGE.trial.pred.json"

python main.py \
      --do_htune --save_dir "$models" --summary_logdir "$logs" \
      --train_file "$trainfile" --dev_file "$devfile" \
      --vocab_lang "$LANGUAGE" --vocab_subdir "$vocabdset" --vocab_size 8000 \
      --device "cuda" --epochs 2 \
      --model "revdict-base" --input_key "gloss" --output_key "$EMBBEDING" \
      --settings "revdict-base-mse" --n_head 2 --n_layers=2 --emb_size 128 \
      --do_pred --pred_file "$pred_file" --pred_output "$pred_output" --pred_mod "lower" \
      --model_path "$models/best/model.pt"
      # --word_emb "glove" \
