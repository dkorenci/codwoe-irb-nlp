#from atilf_codwoe.code.data import JSONDataset
from data_analysis.dataset_old import JSONDataset
from pathlib import Path
import json
from settings import dataset_folder

def loadDataset(lang='en', subset='train', label=None, subdir=None):
    ''' Create and return JSONDataset object  '''
    f = getFile(lang, subset, label, subdir)
    dset = JSONDataset(f)
    #for item in dset: print(item)
    return dset

def loadDatasetJson(lang='en', subset='train', label=None, subdir=None):
    ''' Load raw Json dataset.  '''
    f = getFile(lang, subset, label, subdir)
    with open(f, 'r') as istr:
        dset = json.load(istr)
    #for item in dset: print(item)
    return dset

def getFile(lang='en', subset='train', label=None, subdir=None):
    ''' File of the specified dataset, as pathlib Path '''
    fdir = Path(dataset_folder)
    if subdir: fdir = fdir / subdir
    label_tag = f'.{label}' if label else ''
    fname = f'{lang}.{subset}{label_tag}.json'
    return fdir / fname

def loadTextDataset(lang='en', subset='train', subdir=None):
    ''' Load dataset as list of texts (strings) '''
    jdset = loadDatasetJson(lang, subset, subdir=subdir)
    return [ item['gloss'] for item in jdset ]

def loadTextAndEmbs(lang='en', subset='train', emb='electra', subdir=None, label=None):
    '''
    :param emb: electra, sgns, or char
    :return: list of (gloss (string), vector (ndarray)) pairs
    '''
    import numpy as np
    dset = loadDatasetJson(lang, subset, label, subdir)
    return [(item['gloss'], np.array(item[emb])) for item in dset]

def loadEmbs(lang='en', subset='train', emb='electra', subdir=None, label=None, unique=False):
    '''
    :param emb: electra, sgns, or char
    :return: list of (gloss (string), vector (ndarray)) pairs
    '''
    import numpy as np
    dset = loadDatasetJson(lang, subset, label, subdir)
    list = [np.array(item[emb]) for item in dset]
    result = np.stack(list)
    if unique: result = np.unique(result, axis=0)
    return result

def json_load(fpath):
    '''
    Load json file.
    :return: dict with json data
    '''
    with open(fpath, 'r') as f: dset = json.load(f)
    return dset

def json_write(out_file, json_items):
    ''' Write formatted readable json data to file. '''
    with open(out_file, 'w') as of:
        json.dump(json_items, of, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    #print(loadTextDataset())
    loadTextAndEmbs()


