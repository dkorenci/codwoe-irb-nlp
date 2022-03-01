#     MAIN_COMMAND="${COMMAND} ${TASK} ${ACTION_F} --model \"${MODEL_BASE_NAME}\" --settings \"${MODEL_SETTINGS}\" --train_file \"${DATA_TRAIN}\" --dev_file \"${DATA_DEV}\" --test_file \"${DATA_TEST}\" --device \"cuda:${CUDA}\" --spm_model_path \"${SPM_PATH}\" --save_dir \"${SAVE_DIR}\" --pred_file \"${PRED_FILE}\" --summary_logdir \"${LOG_DIR}\" ${MODEL_PATH_F} ${EMBED_F} ${INPUT_KEY_F} ${OUTPUT_KEY_F} ${EXPERIMENT_ID_F} ${PARAMETERS_F} ${EPOCHS_F} ${HEADS_F} ${LAYERS_F}"
REPO_ROOT=`readlink -f ..`
export PYTHONPATH="$REPO_ROOT:$PYTHONOATH"
echo $PYTHONPATH
#   --pred_file "/datafast/codwoe/train-data_all/en.dev.json" \

PRED_FILE="/datafast/codwoe/trial-data_all/en.trial.complete.json"
vocabdset="orig"

python main.py \
      --do_pred --pred_file "$PRED_FILE" \
      --pred_output "en.trial.pred.json" \
      --vocab_lang "en" --vocab_subdir "$vocabdset" --vocab_size 8000 \
      --device "cuda" \
      --model_path "models/best/model.pt" \
      --settings "revdict-base-mse" --input_key "gloss" --output_key "sgns" \




