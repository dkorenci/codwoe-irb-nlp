# add repo to pythonpath
REPO_ROOT=`readlink -f ../..`
export PYTHONPATH="$REPO_ROOT:$PYTHONOATH"

LANG="it"
#SUBF="/home/damir/Dropbox/projekti/semeval2022/submissions/defmod/submitV4/submitV4-lstm-gru-allvec/defmod.$LANG.json"
SUBF="/home/damir/Dropbox/projekti/semeval2022/submissions/defmod/submitV4/submitV4-gru-allvec-sgns/defmod.$LANG.json"
REFF="/datafast/codwoe/reference_data/$LANG.test.defmod.complete.json"

python3 defmod_score.py --lang $LANG --submission_file "$SUBF" --reference_file "$REFF" \
        --dset_output_file "$LANG.modelout.alldata.json" "/some/path"