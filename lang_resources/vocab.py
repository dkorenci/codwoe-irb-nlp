from sentencepiece import SentencePieceProcessor
from torch import Tensor

# basic, sentencepiece-compatible special tokens
BOS, BOS_ix = "<seq>", 0
EOS, EOS_ix = "</seq>", 1
PAD, PAD_ix = "<pad/>", 2
UNK, UNK_ix = "<unk/>", 3

# additional special tokens for other uses, ie. transformer functionality
MASK = '<mask>'
CLS = '<cls>'
SEP = '<sep>'
PREDICT = '<pred>'
ADDIT_SPECIAL_TOKENS = [MASK, CLS, SEP, PREDICT]

class Vocab():
    ''' Abstract vocabulay class '''

    def encode(self, text):
        ''' Encode string to integer indices. '''
        raise NotImplementedError()

    def decode(self, ind):
        ''' Decode list of integer token indices to string, remove PAD, BOS, and EOS tokens '''
        raise NotImplementedError()

    def maxIndex(self):
        ''' Max integer index of a dictionary item. For tensor init. '''

class SpmVocab(Vocab):
    ''' Vocab based on sentencepiece models, with support for additional special characters '''

    def __init__(self, spm_model):
        '''
        :param spm_model: sentencepiece model, must be build with basic special tokens defined in this file
        '''
        assert isinstance(spm_model, SentencePieceProcessor)
        self._spm_model = spm_model

    def maxIndex(self):
        return self._spm_model.vocab_size()

    def encode(self, text):
        return self._spm_model.encode(text, add_eos=True, add_bos=False)
        #return self._spm_model.encode(text, add_eos=True, add_bos=True)

    def decode(self, ind):
        return self._spm_model.decode(ind)

class PlainVocab(Vocab):
    ''' Vocabulary based on whitespace tokenization and list of tokens. '''
