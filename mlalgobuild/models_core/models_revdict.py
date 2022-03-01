'''
Torch models for reverse dictionary task (mapping gloss text -> word embedding)
'''

import torch
from torch import nn as nn
from torch.nn import functional as F

from lang_resources import vocab
from mlalgobuild.models_core.models_common import PositionalEncoding


class RevdictBase(nn.Module):
    """A transformer architecture for Definition Modeling."""

    def __init__(
        self, max_vocab_idx, d_emb=256, d_output=256, n_head=4, n_layers=4, dropout=0.3, maxlen=512,
            word_emb = None, pad=vocab.PAD_ix, eos=vocab.EOS_ix,
    ):
        super(RevdictBase, self).__init__()
        self.name = str(type(self).__name__)
        self.d_output = d_output
        self.d_emb = d_emb
        #self.d_model = d_model
        self.padding_idx = pad
        self.eos_idx = eos
        self.maxlen = maxlen

        if word_emb is None:
            self.embedding = nn.Embedding(max_vocab_idx, d_emb, padding_idx=self.padding_idx)
        else:
            embs = torch.tensor(word_emb, dtype=torch.float)
            assert (embs.shape[0], embs.shape[1]) == (max_vocab_idx, d_emb) ,\
                    "shape of the pretrained embeddings do not match model dimensions"
            self.embedding = nn.Embedding.from_pretrained(embs, freeze=False,
                                                          padding_idx=self.padding_idx)

        self.positional_encoding = PositionalEncoding(
            d_emb, dropout=dropout, max_len=maxlen
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_emb, nhead=n_head, dropout=dropout, dim_feedforward=d_emb * 2
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.dropout = nn.Dropout(p=dropout)
        self.e_proj = nn.Linear(d_emb, d_output)
        for name, param in self.named_parameters():
            # do not initialize embeddings if pretrained values are used for init
            if "embedding" in name and word_emb is not None: continue
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            else:  # gain parameters of the layer norm
                nn.init.ones_(param)

    def numParameters(self):
        ''' Return the number of model parameters '''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, gloss_tensor):
        src_key_padding_mask = gloss_tensor == self.padding_idx
        embs = self.embedding(gloss_tensor)
        src = self.positional_encoding(embs)
        transformer_output = self.dropout(
            self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask.t())
        )
        #OLD CODE, WITHOUT AVERAGING
        #summed_embs = transformer_output.masked_fill(
        #    src_key_padding_mask.unsqueeze(-1), 0
        #).sum(dim=0)
        #return self.e_proj(F.relu(summed_embs))
        # PERFORM AVERAGING
        masked_out = transformer_output.masked_fill(src_key_padding_mask.unsqueeze(-1), 0)
        averager = gloss_tensor != self.padding_idx
        mask_mean = masked_out.sum(0) / averager.sum(0).unsqueeze(-1)
        return self.e_proj(F.relu(mask_mean))

    @staticmethod
    def load(file):
        return torch.load(file)

    def save(self, file):
        torch.save(self, file)