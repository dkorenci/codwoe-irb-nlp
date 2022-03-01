# add code to pythonpath
REPO_ROOT=`readlink -f ..`
export PYTHONPATH="$REPO_ROOT:$PYTHONOATH"
echo $PYTHONPATH

# Modelbuild parameters
# read from cmdline params, if empty assign default values
LANGS="${1:-en es fr it ru}"
EMBEDS="${2:-sgns electra char}"
ALLVECMOD="${3:-adapt}"
DEVICE="${4:-cuda}"
SUBDIR="${5:-}"
DROP="${6:-0.3}"
IN_DROP="${7:-0.1}"
SCHED="${8:-plat}"
LR="${9:-0.001}"
ARCH="${10:-gru}"
WARMUP="${11:-0.1}"

# input and output folders
## BEA
DATASET_DIR="/home/dkorencic/codwoe-data/dataset/"
TRAIN_OUT_DIR="/home/dkorencic/codwoe-out/defmod/"
#TRAIN_OUT_DIR="/home/dkorencic/snas/codwoe-out/defmod/"
TEST_FILES_DIR="/home/dkorencic/snas/codwoe-data/test/defmod/"
PREDICTIONS_FOLDER="/home/dkorencic/codwoe-out/defmod/combined/"
#PREDICTIONS_FOLDER="/home/dkorencic/snas/codwoe-out/defmod/combined/"
# set to empty string for no trial evaluation
TRIAL_DATA="/home/dkorencic/snas/codwoe-data/trial-data_all/"

# B52
#DATASET_DIR="/datafast/codwoe/dataset/"
#TRAIN_OUT_DIR="/datafast/codwoe/output/defmod/"
#TEST_FILES_DIR="/datafast/codwoe/test_files/defmod/"
#PREDICTIONS_FOLDER="/datafast/codwoe/output/defmod/combined/"
## set to empty string for no trial evaluation
#TRIAL_DATA="/datafast/codwoe/trial-data_all/"

TRAIN_OUT_DIR="$TRAIN_OUT_DIR$SUBDIR"
PREDICTIONS_FOLDER="$PREDICTIONS_FOLDER$SUBDIR"

# dataset version
TRAINDEVDSET="dset_v1"
vocabdset=$TRAINDEVDSET

EPOCHS=450
MAXLEN=64

echo "RUNNING FOR: $LANGS, $EMBEDS, ON $DEVICE"

#TRAIN
for lang in $LANGS; do
  for emb in $EMBEDS; do
      trainfile="$DATASET_DIR$TRAINDEVDSET/$lang.train.json"
      devfile="$DATASET_DIR$TRAINDEVDSET/$lang.dev.json"
      [ ${emb} = "allvec" ] && allvectag=".$ALLVECMOD" || allvectag=""
      modeldir="$TRAIN_OUT_DIR$lang.$emb$allvectag.$ARCH"
      mkdir -p "$modeldir"
      savedir="$modeldir/models"
      logdir="$modeldir/logs"
      outfile="$modeldir/train.out.txt"
      [ ${lang} = "en" ] && vocabsize=8000 || vocabsize=8500
      python main.py \
      --do_train --save_dir "$savedir" --summary_logdir "$logdir" \
      --train_file "$trainfile" --dev_file "$devfile" \
      --vocab_lang ${lang} --vocab_subdir ${vocabdset} --vocab_size ${vocabsize} \
      --device ${DEVICE} --epochs ${EPOCHS} --dropout ${DROP} --in_dropout ${IN_DROP} \
      --learn_rate ${LR} --scheudle ${SCHED} --warmup ${WARMUP} \
      --model "defmod-rnn" --rnn_arch ${ARCH} --input_key ${emb} --allvec_mode ${ALLVECMOD} --output_key "gloss" \
      --settings "defmod-base-xen" --n_layers=2 \
      --word_emb "glove" --emb_size 256 --maxlen ${MAXLEN} > $outfile 2>&1
  done
done

# PREDICT
for lang in $LANGS; do
  for emb in $EMBEDS; do
      trainfile="$DATASET_DIR$TRAINDEVDSET/$lang.train.json"
      devfile="$DATASET_DIR$TRAINDEVDSET/$lang.dev.json"
      [ ${emb} = "allvec" ] && allvectag=".$ALLVECMOD" || allvectag=""
      modeldir="$TRAIN_OUT_DIR$lang.$emb$allvectag.$ARCH"
      mkdir -p "$modeldir"
      savedir="$modeldir/models"
      modelpath="$savedir/best/model.pt"
      predinfile="$TEST_FILES_DIR$lang.test.defmod.json"
      predoutfile="$modeldir/defmod.$lang.$emb$allvectag.json"
      outfile="$modeldir/pred.out.txt"
      [ ${lang} = "en" ] && vocabsize=8000 || vocabsize=8500
      python main.py \
      --do_pred --modout "dsetv1" --pred_file "$predinfile" --pred_output "$predoutfile" \
      --vocab_lang ${lang} --vocab_subdir ${vocabdset} --vocab_size ${vocabsize} --maxlen ${MAXLEN}  \
      --settings "defmod-base-xen" --input_key ${emb} --output_key "gloss" \
      --device ${DEVICE} \
      --model_path "$modelpath" > $outfile 2>&1
      mkdir -p "$PREDICTIONS_FOLDER"
      cp "$predoutfile" "$PREDICTIONS_FOLDER"

      # --do_pred --model_path "models/best/model.pt" --pred_file "$PRED_FILE" --pred_output "en.trial.pred.json" \
      # --modout "dsetv1" --pred_out_align True \

      if [[ "$TRIAL_DATA" != "" ]]; then
          predinfile="$TRIAL_DATA$lang.trial.complete.json"
          predoutfile="$modeldir/defmod.$lang.$emb$allvectag.trial.json"
          python main.py \
            --do_pred --modout "dsetv1" --pred_file "$predinfile" --pred_output "$predoutfile" \
            --pred_out_align True \
            --vocab_lang ${lang} --vocab_subdir ${vocabdset} --vocab_size ${vocabsize} --maxlen ${MAXLEN}  \
            --settings "defmod-base-xen" --input_key ${emb} --output_key "gloss" \
            --device ${DEVICE} \
            --model_path "$modelpath" > $outfile 2>&1
          mkdir -p "$PREDICTIONS_FOLDER"
          cp "$predoutfile" "$PREDICTIONS_FOLDER"
          predoutalign="$modeldir/defmod.$lang.$emb$allvectag.trial.align.txt"
          cp "$predoutalign" "$PREDICTIONS_FOLDER"
      fi
  done
done
