from data_analysis.data_utils import loadDataset, loadTextDataset, loadDatasetJson, getFile
from lang_resources.sentencepice_build import createLoadSpm
from lang_resources.vocab import SpmVocab

import numpy as np
import random

def datasetStatistics(dset, vocab=None):
    num_gloss = len(dset)
    dict_size = len(dset.vocab) if vocab is None else vocab.maxIndex()
    num_tokens = 0
    gloss_lens = []
    for i, item in enumerate(dset):
        gloss = item['gloss']
        tokens = gloss.split() if vocab is None else vocab.encode(gloss)
        gloss_lens.append(len(tokens))
        num_tokens += len(tokens)
    print(f'num. glosses: {num_gloss}; dict.size: {dict_size}; num.tokens: {num_tokens}')
    mn, q25, med, q75, mx, mean, std = statSummary(gloss_lens)
    print(f'gloss size: mean: {mean:.2f}, std: {std:.2f};  min {mn}, q25 {q25}, med {med}, q75 {q75}, max {mx}')

def datasetGlossSample(dset, N=100, seed=28871):
    ''' Print a sample of N glosses.'''
    glosses = [ix['gloss'] for ix in dset]
    random.seed(seed)
    sample = random.sample(glosses, N)
    for g in sample: print(g)

def statSummary(vals):
    mn, mx = np.min(vals), np.max(vals)
    p = np.percentile(vals, [25, 50, 75])
    q25, med, q75 = p[0], p[1], p[2]
    mean = np.mean(vals)
    std = np.std(vals)
    return mn, q25, med, q75, mx, mean, std

def allDsetStats(subdir, label=None):
    for lang in ['en', 'es', 'fr', 'it', 'ru']:
        for subset in ['train', 'dev']:
            print(f'DATASET {lang}.{subset}')
            datasetStatistics(loadDataset(lang, subset, subdir=subdir, label=label))
            print()

def allDsetStatsSpm(subdir, label=None, vocab_size=8000, vocab_subset='train'):
    for lang in ['en', 'es', 'fr', 'it', 'ru']:
        vocab = createLoadSpm(lang=lang, subset=vocab_subset, subdir=subdir, dict_size=vocab_size)
        vocab = SpmVocab(vocab)
        for subset in ['train', 'dev']:
            print(f'DATASET {lang}.{subset}')
            datasetStatistics(loadDataset(lang, subset, subdir=subdir, label=label), vocab)
            print()

def analyzeDuplicates(lang='en', subset='train', subdir='orig', fullPrint=False):
    dset = loadDatasetJson(lang, subset, subdir=subdir)
    txt2ind = {}
    for i, item in enumerate(dset):
        txt = item['gloss']
        if txt in txt2ind: txt2ind[txt].append(i)
        else: txt2ind[txt] = [i]
    print(f'Dataset: {lang}.{subset}')
    nall, nuniq = len(dset), len(txt2ind)
    print(f' unique texts: {nuniq}, all texts: {nall}, difference: {nall-nuniq}')
    if fullPrint:
        txtInd = [(txt, inds) for txt, inds in txt2ind.items()]
        txtInd.sort(key=lambda x: len(x[1]), reverse=True)
        for txt, inds in txtInd: print(inds, txt)

def analyzeDuplicatesOrg(langs=['en', 'fr', 'ru'], subset=['gloss', 'sgns']):
    ''' Modified organizers' code from Discord '''
    import pandas as pd
    for lang in langs:
        df = pd.io.json.read_json(path_or_buf=getFile(lang), orient="records")
        # patch to hashable
        df["sgns"] = df.sgns.apply(tuple)
        df["char"] = df.char.apply(tuple)
        if lang in ['en', 'fr', 'ru']:
            df["electra"] = df.electra.apply(tuple)
        len_df = len(df)
        len_df_dedup = len(df.drop_duplicates(subset=subset))
        print(f'lang: {lang}, subset: [{";".join(subset)}],  len:{len_df}, len-ded:{len_df_dedup}, diff:{len_df_dedup - len_df}')

def allDuplicates():
    for lang in ['en', 'es', 'fr', 'it', 'ru']:
        for subset in ['train', 'dev']:
            analyzeDuplicates(lang, subset, fullPrint=False)

def allDuplicatesOrg():
    analyzeDuplicatesOrg(langs=['es', 'it'], subset=['gloss', 'sgns'])
    analyzeDuplicatesOrg(langs=['es', 'it'], subset=['gloss', 'char'])
    analyzeDuplicatesOrg(langs=['es', 'it'], subset=['gloss', 'sgns', 'char'])
    print()
    analyzeDuplicatesOrg(langs=['en', 'fr', 'ru'], subset=['gloss', 'sgns'])
    analyzeDuplicatesOrg(langs=['en', 'fr', 'ru'], subset=['gloss', 'sgns', 'electra'])

if __name__ == '__main__':
    #loadDataset()
    #allDsetStats(subdir='orig')
    allDsetStatsSpm(subdir='dset_v1')
    #datasetGlossSample(loadDataset('ru', 'train'), 500)
    #analyzeDuplicates('en', 'dev', fullPrint=True)
    #analyzeDuplicatesOrg(subset=['gloss'])
    #allDuplicates()
    #allDuplicatesOrg()