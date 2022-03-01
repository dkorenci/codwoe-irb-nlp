import json
from copy import deepcopy
from pathlib import Path
import re

from data_analysis.data_utils import *
from data_analysis.basic_stats import datasetGlossSample
from settings import dataset_folder

def splitGloss(gls):
    '''
    Split gloss (a string of whitespace-separated tokens) into sub-glosses,
    separated by ';' character.
    :param gls: string gloss
    :return: list of string sub-glosses
    '''
    subgls = [sg.strip() for sg in gls.split(';')]
    # filter out empty string (';' can occur at start or end)
    subgls = [sg for sg in subgls if sg != '']
    return subgls

def removeBorderPunctWs(gls):
    '''Remove whitespace and punctuation from start or end of the gloss'''
    gls = gls.strip()
    if gls.endswith('.'): gls = gls[:-1]
    if gls.startswith(':'): gls = gls[1:]
    return gls.strip()

def extractRemoveLabelsFrIt(gls):
    '''
    Extract/remove labels for french and italian.
    Labels are at gloss start, in format: ( label1 ) [[,] ( labelK )] +
    :param gls:
    :return: list of lowercased labels, gloss without labels
    '''
    tokens = gls.split()
    labels_exist = tokens[0] == '('
    if not labels_exist: return [], gls
    # extract labels
    inside_lbl = False; label = None; labels = []
    def_start = None
    for i, tok in enumerate(tokens):
        if not inside_lbl:
            if tok == '(':
                inside_lbl = True
            elif tok == ')':
                #continue # ignore such rare errors
                raise Exception(f'misformed label part: {gls}')
            elif tok == ',': continue # ignore commas
            else: def_start = i; break # end of labels part
        else:
            if tok == ')':
                if label is None: raise Exception(f'parenteses closed without label inside: {gls}')
                labels.append(label)
                label = None; inside_lbl = False
            elif tok == '(': raise Exception(f'multiple consequent open paranteses: {gls}')
            else:
                if label is not None: # enable multi-token labels
                    label = ' '.join([label, tok])
                    #raise Exception(f'multiple label tokens: {gls}')
                else: label = tok
    labels = [l.lower() for l in labels]
    stripped_gls = ' '.join(tokens[def_start:])
    return labels, stripped_gls

def extractRemoveLabelsEs(gls):
    '''
    Extract and remove laels from spanish glosses.
    Labels are formated as: label[ y labelK]+[.]|
    '''
    # fullglserr = [
    #     'acción o efecto de habilitar',
    #     'cuerpos o deidades individuales en una sola masa',
    #     'de transmutación en cuerpo animal seria la quimera',
    #     'desde la fundación del estado de israel',
    #     'mezcla de uno o varios conjuntos de unidades de objetos',
    #     'separación o independización de una nación de parte de su pueblo o de su territorio'
    # ]
    # for e in fullglserr:
    #     if e in gls.lower():
    #         raise Exception(f'parsing err: {gls}')
    labels_exist = '|' in gls
    if not labels_exist: return [], gls
    if gls.count('|') > 1: return [], gls # rare error, misformed gloss
    parts = gls.split('|')
    labels, gls_fin = parts[0], parts[1]
    if len(labels) > len(gls_fin)*3:
        # very long label, indicates misformed gloss labels, ie. label part is probably gloss
        if gls_fin.strip() != '': # empty gloss case ignored
            #raise Exception(f'gloss err: {gls}')
            return [], gls
        # else: empty gloss -> extract labels, leave handling to client code
    gls_fin = gls_fin.strip()
    # handle empty gloss in client code
    #if len(gls_fin) == 0: raise Exception(f'empty gloss: {gls}')
    if labels == '': return [], gls_fin
    if labels[-1] == '.': labels = labels[:-1] # remove dot from the end
    labels = labels.lower()
    splitter = re.compile(r"[,\.\(\)]| y | e ")
    labels = [l.strip() for l in splitter.split(labels)]
    def stripPrefix(s, prfx):
        for p in prfx:
            if s.startswith(p): return s[len(p):]
        return s
    labels = [stripPrefix(l, ['en la ', 'en ']) for l in labels]
    labels = [l for l in labels if l != '']
    return labels, gls_fin

def extractRemoveLabelsRu(gls):
    '''
    Extract and remove laels from russian glosses.
    Labels are formated as: [label\w?.]+,? ,
    for example: 'церк. уменьш.-ласк .', 'устар. , диал. , Сиб. , Якут. , Амур . '
    '''
    tokens = gls.split()
    if len(tokens) == 0 or len(tokens) == 1: # empty gloss or single word
        return [], gls
    labels_exist = tokens[0].endswith('.') or tokens[1] == '.'
    if not labels_exist: return [], gls
    labels = []; start_gls_idx = None
    for i, tok in enumerate(tokens):
        if tok in ['.', ',']: continue
        elif tok.endswith('.'): labels.append(tok[:-1])
        else:
            if i < len(tokens)-1 and tokens[i+1] == '.':
                labels.append(tok)
            else:
                start_gls_idx = i;
                break;
    if start_gls_idx is None: # entire gloss consists of labels -> error
        start_gls_idx = len(tokens)
    labels = [l.lower().strip() for l in labels if l.strip() != '']
    gloss = ' '.join(tokens[start_gls_idx:])
    gloss = gloss.strip()
    return labels, gloss

def exctractRemoveLabels(gls, lang):
    '''
    Exctract label string from the start of the gloss, create clean gloss
    :param gls: string gloss
    :param lang: language code
    :return: list of lowercased string labels, gloss without labels
    '''
    if lang == 'en': # english glosses have no labels
        return [], gls
    elif lang == 'fr' or lang == 'it': return extractRemoveLabelsFrIt(gls)
    elif lang == 'es': return extractRemoveLabelsEs(gls)
    elif lang == 'ru': return extractRemoveLabelsRu(gls)
    else: raise ValueError(f'language not supported: {lang}')

def extractLabelsTest():
    print(extractRemoveLabelsFrIt('this is gloss txt . '))
    print(extractRemoveLabelsFrIt(' ( label1 ) this is gloss txt . '))
    print(extractRemoveLabelsFrIt(' ( label1 ) ( label2 ) this is gloss txt . '))
    print(extractRemoveLabelsFrIt(' ( label1 ) , ( label2 ) this is gloss txt . '))

def extractLabelsDsetTest(lang, subset, print_every=200):
    ''' Test on all glosses in a datasset. '''
    import random, traceback
    dset = loadTextDataset(lang, subset); N = len(dset)
    labels = []; num_lab = 0; num_err = 0; num_empty = 0
    for gls in dset:
        try:
            labs, gls_nolab = exctractRemoveLabels(gls, lang)
            if gls_nolab.strip() == '': num_empty += 1
        except Exception as e:
            num_err += 1
            print(f'ERROR: {e} \n')
            #print(traceback.format_exc())
            labs, gls_nolab = [], gls
        labels.extend(labs)
        if len(labs) > 0: num_lab += 1
        if random.random() < 1.0/print_every:
            print(f'gloss: {gls}')
            print(f'labels: {labs}')
            print(f'gloss no labels: {gls_nolab}')
            print()
    print('all labels:')
    for l in sorted(set(labels)): print(l)
    print()
    print(f'num glosses {N}, num labeled glosses {num_lab}, {num_lab/N*100:.3}%')
    print(f'num errrors: {num_err}, {num_err/N*100:.3}%')
    print(f'num empty gosses: {num_empty}, {num_empty / N * 100:.3}%')

def lowercase(gls): return gls.lower()

def splitAttribGlosses(items):
    '''
    Split each gloss in the dataset, and assign to each sub-gloss the
    attributes of the original gloss.
    :param items: list of dict items, each having a 'gloss' attribute and possibly other attributes
    :return: list of dict items with sub-glosses and corresponding (super-gloss) attributes
    '''
    result = []
    for itm in items:
        orig_id = itm['id']
        splits = splitGloss(itm['gloss']); num_splits = len(splits)
        for ix, subgls in enumerate(splits):
            subitm = deepcopy(itm)
            subitm['gloss'] = subgls
            subitm['id'] = f'{orig_id}.{ix+1}' if num_splits > 1 else orig_id
            result.append(subitm)
    return result

def extractLabels(items, lang):
    '''
    Extract and remove labels from the start of the glosses, add labels as a new attribute.
    Items are modified in place and returned.
    :param items: list of dict items, each having a 'gloss' attribute and possibly other attributes
    :return: list of dict items with modified glosses and new 'label' attribute
    '''
    num_err = 0
    for itm in items:
        gloss = itm['gloss']
        try:
            labels, new_gls = exctractRemoveLabels(gloss, lang)
        except Exception as e:
            num_err += 1
            print(f'ERROR: {e} \n')
            labels, new_gls = [], gloss
        if new_gls == '':
            print(f'empty gloss after lab.extract, orig gloss: "{gloss}"')
        labels = ']['.join(labels)
        itm['gloss'] = new_gls
        assert 'labels' not in itm
        itm['labels'] = labels
    print(f'num. extraction errors: {num_err}')
    return items

def splitGlossTests():
    print(splitGloss('no semicolon here'))
    print(splitGloss('one gloss ; two gloss ; three .'))

def removeEmptyGlosses(items):
    new_items = []; num_empty = 0
    for itm in items:
        if itm['gloss'].strip() != '': new_items.append(itm)
        else: num_empty += 1
    print(f'num. empty glosses: {num_empty}\n')
    return new_items

def normalizeGlosses(items, punct=True, lower=True):
    '''
    Lowercase, remove punctuation and whitespace from start and end.
    :param items:
    :return: items with modified glosses
    '''
    for ix, itm in enumerate(items):
        gls = itm['gloss']
        if punct: gls = removeBorderPunctWs(gls)
        if lower: gls = lowercase(gls)
        itm['gloss'] = gls
    return items

def transformDataset(lang, subset, label, sub_folder=None, punct=True, lower=True, split=True, labels=True):
    '''
    Load json dataset, transform glosses, and save as json.
    :param lang: language, ex. 'en'
    :param subset: 'dev' or 'test'
    :param sub_folder: subfolder of settings.dataset_folder to save output in
    :param label: output filename is lang.subset.label.json
    :param punct: weather to remove punctuation
    :param lower: weather to lowercase
    :param split: weather to split glosses
    :param labels: weather to extract labels from glosses
    :return:
    '''
    json_items = loadDatasetJson(lang, subset, subdir='orig')#[:200]
    save_folder = Path(dataset_folder)
    if sub_folder:
        save_folder = save_folder / sub_folder
        save_folder.mkdir(exist_ok=True)
    json_items = removeEmptyGlosses(json_items)
    if labels: json_items = extractLabels(json_items, lang)
    json_items = removeEmptyGlosses(json_items)
    if split: json_items = splitAttribGlosses(json_items)
    json_items = normalizeGlosses(json_items, punct, lower)
    out_file = f'{lang}.{subset}.{label}.json' if label else f'{lang}.{subset}.json'
    out_file = save_folder / out_file
    with open(out_file, 'w') as of:
        json.dump(json_items, of, ensure_ascii=False, indent=2)

def reformatDataset(lang, subset, label, sub_folder=None):
    '''
    Rewrite dataset json files in a more readable formatting.
    '''
    json_items = loadDatasetJson(lang, subset)#[:200]
    save_folder = Path(dataset_folder)
    if sub_folder:
        save_folder = save_folder / sub_folder
        save_folder.mkdir(exist_ok=True)
    out_file = f'{lang}.{subset}.{label}.json' if label else f'{lang}.{subset}.json'
    out_file = save_folder / out_file
    with open(out_file, 'w') as of:
        json.dump(json_items, of, ensure_ascii=False, indent=2)

def dsetTransformTest(lang, subset):
    transformDataset(lang, subset, 'slp', 'test')
    dset = loadDatasetJson(lang, subset, 'slp', 'test')
    datasetGlossSample(dset, 100)

def reformattAll():
    for lang in ['en', 'fr', 'it', 'es', 'ru']:
        for subset in ['dev', 'train']:
            reformatDataset(lang, subset, label=None, sub_folder='orig_reformatted')

def dsetV1Modout(gloss):
    '''
    Produces the output version of a gloss to be more compatible
    with original format, in view of the transformations made to produce datasetV1
    :param gloss: str
    :return: transformed gloss as str
    '''
    gloss = gloss.strip()
    tokens = gloss.split()
    if len(tokens) <= 1: return gloss
    # first letter 2 upper, add dot @ end
    gloss = gloss[0].upper() + gloss[1:] + ' .'
    return gloss

def createDsetV1():
    for lang in ['en', 'fr', 'it', 'es', 'ru']:
        for subset in ['dev', 'train']:
            print(f'--------------- TRANSFORMING {lang}{subset} ---------------\n')
            transformDataset(lang, subset, '', 'dset_v1')
            dset = loadDatasetJson(lang, subset, '', 'dset_v1')
            datasetGlossSample(dset, 100)
            print(f'--------------- TRANSFORMING {lang}{subset} FINISHED ---------------\n\n')

def createDsetOrigLcase():
    for lang in ['en', 'fr', 'it', 'es', 'ru']:
        for subset in ['dev', 'train']:
            print(f'--------------- TRANSFORMING {lang}{subset} ---------------\n')
            transformDataset(lang, subset, label=None, sub_folder='orig_lc',
                             punct=False, lower=True, split=False, labels=False)
            dset = loadDatasetJson(lang, subset, '', 'orig_lc')
            datasetGlossSample(dset, 100)
            print(f'--------------- TRANSFORMING {lang}{subset} FINISHED ---------------\n\n')

if __name__ == '__main__':
    #splitGlossTests()
    #dsetTransformTest('es', 'dev')
    #extractLabelsTest()
    #extractLabelsDsetTest('ru', 'train')
    #createDsetV1()
    #reformattAll()
    createDsetOrigLcase()