#SUBDIR="submitV3/"
#SUBDIR="submitV4/"
DROP=0.3
IN_DROP=0.1
SCHED=plat
LR=0.001
#ARCH=lstm

# SUBMIT V4 - 450 epochs
SUBDIR="submitV4-lstm/"
./defmod_batch.sh "en fr" "allvec" "concat" "cuda:1" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} "lstm" &
./defmod_batch.sh "it ru" "allvec" "concat" "cuda:1" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} "lstm" &
./defmod_batch.sh "es" "allvec" "concat" "cuda:1" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} "lstm" &
#./defmod_batch.sh "en fr it" "sgns" "concat" "cuda:0" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} "lstm" &
#./defmod_batch.sh "es ru" "sgns" "concat" "cuda:0" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} "lstm" &

SUBDIR="submitV4-gru/"
./defmod_batch.sh "en fr" "allvec" "concat" "cuda:2" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} "gru" &
./defmod_batch.sh "it ru" "allvec" "concat" "cuda:2" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} "gru" &
./defmod_batch.sh "es" "allvec" "concat" "cuda:2" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} "gru" &
#./defmod_batch.sh "en fr it" "sgns" "concat" "cuda:0" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} "gru" &
#./defmod_batch.sh "es ru" "sgns" "concat" "cuda:0" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} "gru" &

# long dset test
#DSET=orig_lc
#./defmod_batch_lc.sh "en" "allvec" "concat" "cuda:0" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} ${ARCH} ${DSET}&


# SUBMIT V3

#./defmod_batch.sh "en es" "sgns" "concat" "cuda:1" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} ${ARCH} &
#./defmod_batch.sh "fr ru" "sgns" "concat" "cuda:2" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} ${ARCH} &
#./defmod_batch.sh "it" "sgns" "concat" "cuda:3" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} ${ARCH} &

#./defmod_batch.sh "en" "allvec" "concat" "cuda:0" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} ${ARCH} &
#./defmod_batch.sh "es" "allvec" "concat" "cuda:1" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} ${ARCH} &
#./defmod_batch.sh "fr" "allvec" "concat" "cuda:1" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} ${ARCH} &
#./defmod_batch.sh "it" "allvec" "concat" "cuda:2" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} ${ARCH} &
#./defmod_batch.sh "ru" "allvec" "concat" "cuda:2" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} ${ARCH} &

#./defmod_batch.sh "en es fr it ru" "sgns" "adapt" "cuda:1" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} ${ARCH} &

# SUBMIT V2
#./defmod_batch.sh "en es" "allvec" "concat" "cuda:1" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} &
#./defmod_batch.sh "en es" "sgns" "adapt" "cuda:1" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} &
#
#./defmod_batch.sh "fr" "allvec" "concat" "cuda:0" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} &
#./defmod_batch.sh "fr" "sgns" "adapt" "cuda:0" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} &
#
#./defmod_batch.sh "it ru" "allvec" "concat" "cuda:2" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} &
#./defmod_batch.sh "it ru" "sgns" "adapt" "cuda:2" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} &


#
#./defmod_batch.sh "en" "allvec" "merge" "cuda:2" ${SUBDIR} ${DROP} ${IN_DROP} ${SCHED} ${LR} &
#./defmod_batch.sh "es" "sgns electra char" "cuda:1" &
#./defmod_batch.sh "fr" "sgns electra char" "cuda:1" &
#./defmod_batch.sh "it" "sgns electra char" "cuda:2" &
#./defmod_batch.sh "ru" "sgns electra char" "cuda:3" &

