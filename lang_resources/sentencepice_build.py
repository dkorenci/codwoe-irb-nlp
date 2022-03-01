import sentencepiece as spm
spm.set_random_generator_seed(1713)

import tempfile

from pathlib import Path

from lang_resources.vocab import *
from settings import lang_resources_folder
from data_analysis.data_utils import loadDatasetJson

def buildSaveSpm(lang, subset, subdir, label, method, dict_size, spm_file):
    # modified from atilf_codwoe repo code
    dset = loadDatasetJson(lang, subset, subdir=subdir, label=label)
    with tempfile.NamedTemporaryFile(mode='w+') as temp_fp:
        for gls in (itm['gloss'] for itm in dset): print(gls, file=temp_fp)
        temp_fp.seek(0)
        spm.SentencePieceTrainer.train(
            input=temp_fp.name, model_type=method, model_prefix=spm_file,
            vocab_size=dict_size, pad_id=PAD_ix, pad_piece=PAD, eos_id=EOS_ix, eos_piece=EOS,
            bos_id=BOS_ix, bos_piece=BOS, unk_id=UNK_ix, unk_piece=UNK,
        )

def createLoadSpm(lang, subset, subdir=None, label=None, method='unigram', dict_size=8000):
    ''' Create sentencepiece dict/tokenizer and save it to cache folder,
     or load and return it if it exists. '''
    spm_folder = Path(lang_resources_folder)/'sentencepiece'
    spm_folder.mkdir(exist_ok=True)
    subdirl = f'-subdir[{subdir}]' if subdir else ''
    ll = f'-label[{label}]' if label else ''
    file_label = f'spm{subdirl}-{lang}-{subset}{ll}-{method}-{dict_size}'
    spm_file = spm_folder/file_label
    if not (spm_file.with_suffix('.model').exists() and spm_file.with_suffix('.vocab').exists()):
        buildSaveSpm(lang, subset, subdir, label, method, dict_size, spm_file)
    spm_model = spm.SentencePieceProcessor(model_file=str(spm_file.with_suffix('.model')))
    setattr(spm_model, 'id', file_label)
    return spm_model

def spmDemo(lang, subset, method, txt, dict_size=8000, subdir=None, label=None):
    spm = createLoadSpm(lang, subset, subdir, label, method, dict_size=dict_size)
    print(f'spm length: {len(spm)}')
    print(PAD_ix, EOS_ix, BOS_ix, UNK_ix)
    toks = spm.encode_as_pieces(txt)  # tokenization, to a list of spm tokens
    print(toks)
    idxs = spm.encode(txt) #, add_eos=True, add_bos=True)  # encode txt into a list of (token) indices
    print(idxs)
    print(idxs+[PAD_ix, PAD_ix])
    txtr = spm.decode(idxs+[PAD_ix, PAD_ix])  # reconstruct orig. txt from a list of indices
    print(txtr)
    txtr = spm.decode_pieces(toks)  # reconstruct txt from a list of tokens
    print(txtr)

if __name__ == '__main__':
    #createLoadSpm('en', 'dev', 'unigram', 2000)
    spmDemo('en', 'train', 'unigram', 'do travelers travel if they are given financial incentives',
            subdir='dset_v1')