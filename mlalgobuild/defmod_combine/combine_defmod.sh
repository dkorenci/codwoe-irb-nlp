REPO_ROOT=`readlink -f ../..`
export PYTHONPATH="$REPO_ROOT:$PYTHONOATH"
echo $PYTHONPATH

f1='/home/damir/Dropbox/projekti/semeval2022/submissions/defmod/submitV7/submitV7-gru-allvec-el/'
f2='/home/damir/Dropbox/projekti/semeval2022/submissions/defmod/submitV7/submitV7-gru-electra/'
fout='/home/damir/Dropbox/projekti/semeval2022/submissions/defmod/submitV7/submitV7-gru-allvec-electra/'
mkdir -p "$fout"

python3 combine_defmod_pred.py --folder1 "$f1" --folder2 "$f2" --folderOut "$fout" > "$fout/comb.log.txt"