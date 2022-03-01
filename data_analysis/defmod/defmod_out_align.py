'''
Analyses of defmod models' output bases on reference data.
'''

from pathlib import Path
import random

from data_analysis.data_utils import json_load, json_write

def modelout2refdata(modelout, refdata, rndseed=816223):
    '''
    Output, readable and side by side, out glosses produced by a model and ref. data glosses.
    Output two files in modelout's folder: one with sorted glosses, one with rnd. ordere glosses (for sampling)
    :param modelout: json file with model out
    :param refdata: json file with reference data
    '''
    mout = json_load(modelout)
    id2mout = {it['id']:it for it in mout}
    refd = json_load(refdata)
    id2ref = {it['id']:it for it in refd}
    assert set(id2mout) == set(id2ref), 'item IDs must be the same in both files'
    ids = sorted(list(set(id2mout)), key=lambda id:int(id.split('.')[-1]))
    out_items = []
    for i in ids: # add items in ordered by id
        itm = {'id':i}
        itm['word'] = id2ref[i]['word']
        itm['gloss-ref']=id2ref[i]['gloss']
        itm['gloss-out']=id2mout[i]['gloss']
        itm['example']=id2ref[i]['example']
        out_items.append(itm)
    outfile = Path(modelout).with_suffix('.refalign.json')
    json_write(outfile, out_items)
    # write rnd. shuffled items
    random.seed(rndseed)
    random.shuffle(out_items)
    outfile = Path(modelout).with_suffix('.refalign.rndshuff.json')
    json_write(outfile, out_items)

if __name__ == '__main__':
    modelout2refdata('/home/damir/Dropbox/projekti/semeval2022/submissions/defmod/submitV5/defmod.en.sgns.json',
                     '/datafast/codwoe/reference_data/en.test.defmod.complete.json')
    #modelout2refdata('/home/damir/Dropbox/projekti/semeval2022/submissions/defmod/submitV4/combined-lstm-gru/defmod.en.json',
    #                 '/datafast/codwoe/reference_data/en.test.defmod.complete.json')