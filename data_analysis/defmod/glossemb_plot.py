'''
Plot of 2D projections of gloss embeddings, contrasting the train and test semantics.
'''

import numpy as np
from numpy.random import choice
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from umap import UMAP

from data_analysis.data_utils import loadEmbs

def plotEmbeddings(dset='dset_v1', lang='en', emb='sgns', reduce_method='TSNE',
                   rndSeed=81773, subsample=None):
    test_embs = loadEmbs(lang, subset='test', emb=emb, subdir='defmod-test', label='defmod')
    train_embs = loadEmbs(lang, subset='train', emb=emb, subdir=dset)
    test_len, train_len = len(test_embs), len(train_embs)
    if subsample: # sample datasets to subsample% of orig. size
        np.random.seed(rndSeed)
        test_embs = test_embs[choice(test_len, int(test_len*subsample), replace=False), :]
        test_len = int(test_len*subsample)
        train_embs = train_embs[choice(train_len, int(train_len * subsample), replace=False), :]
        train_len = int(train_len * subsample)
    all_embs = np.concatenate((train_embs, test_embs))
    if reduce_method == 'TSNE': dimReduce = TSNE(random_state=rndSeed)
    elif reduce_method == 'UMAP': dimReduce = UMAP(random_state=rndSeed)
    embs2d = dimReduce.fit_transform(all_embs)
    dset_color = np.concatenate((np.ones(train_len)*0.7, np.ones(test_len)*0.3))
    fig, ax = plt.subplots(); #ax.set_aspect('equal')
    ax.scatter(embs2d[:, 0], embs2d[:, 1], s=0.5, c=dset_color,
                alpha=0.3, cmap='RdYlGn')
    figid=f'trainVsTestDensity{reduce_method}-{lang}-{emb}'
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(figid+'.png', dpi=300)
    #plt.show()

def runAllPlots():
    for lang in ['en', 'es', 'fr', 'it', 'ru']:
        vecs = ['sgns', 'electra'] if lang in ['en', 'fr', 'ru'] else ['sgns']
        for vec in vecs:
            print(lang, vec)
            plotEmbeddings(lang=lang, emb=vec, subsample=None, reduce_method='TSNE')

if __name__ == '__main__':
    #plotEmbeddings(lang='en', emb='sgns', subsample=0.1, reduce_method='TSNE')
    runAllPlots()