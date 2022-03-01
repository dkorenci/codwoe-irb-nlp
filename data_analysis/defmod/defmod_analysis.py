'''
Analysis of defmod models's output and scores.
'''
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

from data_analysis.data_utils import loadEmbs, json_load

BEST_SUBMITTED = {lang:f'/datafast/codwoe/defmod-analysis/submitV4-lstm-gru-allvec/{lang}.modelout.alldata.json'
                  for lang in ['en', 'es', 'fr', 'it', 'ru']}
BEST_V4 = {lang:f'/datafast/codwoe/defmod-analysis/submitV4-gru-allvec-sgns/{lang}.modelout.alldata.json'
                  for lang in ['en', 'es', 'fr', 'it', 'ru']}

bestSim = lambda x: x[0]
top10avg = lambda x: np.average(x[:10])
top20avg = lambda x: np.average(x[:20])
top50avg = lambda x: np.average(x[:50])
top10med = lambda x: np.median(x[:10])
top20q75 = lambda x: np.percentile(x[20], 75)

simAgg = {
            'best': bestSim,
            'top10avg': top10avg,
            #'top20avg': top20avg,
            #'top50avg': top50avg,
            'top10med': top10med,
            'top20q75': top20q75
          }

def percentileAgg(perc):
    return lambda x: np.sum(x >= perc)

def correlateScoreVStrainCloseness(modelout, lang='en', train_dset='dset_v1', emb_type='sgns',
                                   closeness='sim'):
    print(f'CORRELATION FOR: {lang}, {emb_type}')
    train_embs = loadEmbs(lang, subset='train', emb=emb_type, subdir=train_dset)
    modelout = json_load(modelout)
    modelout = sorted(modelout, key=lambda r: r["id"])
    test_embs = np.stack([np.array(itm[emb_type]) for itm in modelout])
    cossims = cosine_similarity(test_embs, train_embs)
    cossims = -np.sort(-cossims, axis=1)
    top25cos, top20cos, top10cos = np.percentile(cossims.flatten(), [75, 80, 90])
    for score in ['moverscore', 'sense-BLEU', 'lemma-BLEU']:
        scores = [itm[score] for itm in modelout]
        if closeness == 'sim': aggFuncs = simAgg
        elif closeness == 'percentile':
            aggFuncs = {
                'top25cos': percentileAgg(top25cos),
                'top20cos': percentileAgg(top20cos),
                'top10cos': percentileAgg(top10cos),
            }
        for sim in aggFuncs.keys():
            sima = aggFuncs[sim]
            sims = [ sima(cossims[ix]) for ix in range(len(modelout)) ]
            corr, pval = spearmanr(scores, sims)
            print(f'score {score:10} ; sim {sim:8} : corr={corr:.4f} pval={pval:.4f}')
    print()

def scoreAveragesOnClosnessSubgroups(modelout, lang='en', train_dset='dset_v1', emb_type='sgns',
                                   score='moverscore'):
    print(f'CORRELATION FOR: {lang}, {emb_type}')
    train_embs = loadEmbs(lang, subset='train', emb=emb_type, subdir=train_dset)
    modelout = json_load(modelout)
    modelout = sorted(modelout, key=lambda r: r["id"])
    test_embs = np.stack([np.array(itm[emb_type]) for itm in modelout])
    cossims = cosine_similarity(test_embs, train_embs)
    cossims = -np.sort(-cossims, axis=1)
    NO = len(modelout)
    for score in ['moverscore', 'sense-BLEU', 'lemma-BLEU']:
        scores = [itm[score] for itm in modelout]
        for sim in simAgg.keys():
            sima = simAgg[sim]
            sims = [ sima(cossims[ix]) for ix in range(NO) ]
            for perc in [10, 20, 25]:
                bottom, top = np.percentile(sims, [perc, 100-perc])
                topScore = np.average([scores[ix] for ix in range(NO) if sims[ix] >= top])
                bottomScore = np.average([scores[ix] for ix in range(NO) if sims[ix] <= bottom])
                print(f'score {score:10} ; sim {sim:8} : low{perc}%={bottomScore:.4f} top{perc}%={topScore:.4f}')
    print()

def runAllscoresVsCloseness(lang2outfile):
    for lang in ['en', 'es', 'fr', 'it', 'ru']:
        vecs = ['sgns', 'electra'] if lang in ['en', 'fr', 'ru'] else ['sgns']
        for vec in vecs:
            correlateScoreVStrainCloseness(lang2outfile[lang], lang=lang, train_dset='dset_v1',
                                           emb_type=vec, closeness='percentile')

def runAllSubGroupAvgs(lang2outfile):
    for lang in ['en', 'es', 'fr', 'it', 'ru']:
        vecs = ['sgns', 'electra'] if lang in ['en', 'fr', 'ru'] else ['sgns']
        for vec in vecs:
            scoreAveragesOnClosnessSubgroups(lang2outfile[lang], lang=lang, train_dset='dset_v1', emb_type=vec)

if __name__ == '__main__':
    #scoreAveragesOnClosnessSubgroups(BEST_SUBMITTED['en'], train_dset='dset_v1', score='sense-BLEU')
    #runAllscoresVsCloseness(BEST_SUBMITTED)
    runAllSubGroupAvgs(BEST_SUBMITTED)