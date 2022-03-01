#     MAIN_COMMAND="${COMMAND} ${TASK} ${ACTION_F} --model \"${MODEL_BASE_NAME}\" --settings \"${MODEL_SETTINGS}\" --train_file \"${DATA_TRAIN}\" --dev_file \"${DATA_DEV}\" --test_file \"${DATA_TEST}\" --device \"cuda:${CUDA}\" --spm_model_path \"${SPM_PATH}\" --save_dir \"${SAVE_DIR}\" --pred_file \"${PRED_FILE}\" --summary_logdir \"${LOG_DIR}\" ${MODEL_PATH_F} ${EMBED_F} ${INPUT_KEY_F} ${OUTPUT_KEY_F} ${EXPERIMENT_ID_F} ${PARAMETERS_F} ${EPOCHS_F} ${HEADS_F} ${LAYERS_F}"
REPO_ROOT=`readlink -f ..`
export PYTHONPATH="$REPO_ROOT:$PYTHONOATH"
echo $PYTHONPATH
#   --pred_file "/datafast/codwoe/train-data_all/en.dev.json" \
      #--model_path "models/m2m-baseline/best/model.pt" \
# --vocab_subdir "dset_v1"

PRED_FILE="/datafast/codwoe/trial-data_all/en.trial.complete.json"

# !!! --modout 'dsetv1' \

python main.py \
      --pred_file "$PRED_FILE" \
      --pred_output "en.trial.pred.json" --pred_out_align True \
      --do_pred --modout "dsetv1" --device "cuda:0" \
      --vocab_lang "en" --vocab_subdir "dset_v1" --vocab_size 8000 --maxlen 64  \
      --settings "defmod-base-xen" --input_key "allvec" --output_key "gloss" \
      --model_path "/home/damir/nessieshr/models-merge-200e/best/model.pt"
      #--model_path "models/best/model.pt" \
      #--model_path "/datafast/codwoe/models/defmod-rnn-spm8000-gateact-dsetV1-mlen64-bea-300epochs-allvec/model.pt"
      #--model_path "/datafast/codwoe/models/defmod-en-transf4x4-glove-dset_v1/model.pt"
      #--model_path "/datafast/codwoe/models/defmod-rnn-spm8000-gateact-dsetV1-bea/model.pt"
      #--model_path "/datafast/codwoe/models/defmod-rnn-test-dset_v1/model.pt" \



