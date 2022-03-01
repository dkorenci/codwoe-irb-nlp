'''
Torch models for definition modeling task (mapping word embedding -> gloss text )
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM, GRU
import numpy as np

from lang_resources import vocab
from mlalgobuild.models_core.models_common import PositionalEncoding


class InputAdapterMLP(nn.Module):
    ''' Multilayer perceptron. '''
    def __init__(self, input_size, hidden_size, output_size, n_layers=1,
                 activation=nn.Tanh, in_dropout=0.1, net_dropout=0.4):
        '''
        :param n_layers: number of hidden layers, if 0 adapter is a linear transformation
        '''
        super(InputAdapterMLP, self).__init__()
        assert isinstance(n_layers, int) and n_layers >= 0, f"invalid number of layers: {n_layers}"
        self.in_drop = nn.Dropout(in_dropout)
        self.net_drop = nn.Dropout(net_dropout)
        self.flatten = nn.Flatten()
        if n_layers == 0:
            layers = [nn.Linear(input_size, output_size), activation()]
        else:
            layers = [nn.Linear(input_size, hidden_size), activation(), self.net_drop,
                      nn.Linear(hidden_size, output_size), activation()]
            if n_layers > 1: # insert n_layers-1 hidden2hidden mappings before the last layer
                for i in range(n_layers-1):
                    layers.insert(3, self.net_drop)
                    layers.insert(3, activation())
                    layers.insert(3, nn.Linear(hidden_size, hidden_size))
        self.linear_relu_stack = nn.Sequential(*tuple(layers))

    def forward(self, x):
        x = self.in_drop(x)
        logits = self.linear_relu_stack(x)
        return logits

class DefmodBase(nn.Module):
    ''' Base class for definition modeling models. '''
    def __init__(self, vocab_size, d_emb=256, d_input=256, maxlen=256,
                 word_emb=None, pad=vocab.PAD_ix, eos=vocab.EOS_ix, in_dropout=0.1, net_dropout=0.4):
        '''
        :param vocab_size: max. token index + 1, determined size of the embedding table
        :param d_emb: dimension of the model's word embedding, ie, model dimension
        :param d_input: dimension od the input gloss vector, if != from d_emb
                gloss vector size will be adapted using a MLP model
        :param maxlen: maximum length of an input sequence
        :param word_emb: if None, the matrix with pre-trained embeddings, with dim (vocab_size, d_emb)
        :param pad: padding token index
        :param eos: end-of-sequence token index
        :param in_dropout: dropout for input data - word and gloss embeddings
        :param net_dropout: dropout for network layers
        '''
        super(DefmodBase, self).__init__()
        self.name = str(type(self).__name__)
        self.vocab_size = vocab_size
        self.d_input = d_input
        self.d_emb = d_emb
        self.padding_idx = pad
        self.eos_idx = eos
        self.maxlen = maxlen
        self.word_emb = word_emb
        self.in_dropout, self.net_dropout = in_dropout, net_dropout
        self.in_drop, self.net_drop = nn.Dropout(in_dropout), nn.Dropout(net_dropout)
        # setup gloss vector adaptation to model size
        if d_emb != d_input:
            self.input_adapt = InputAdapterMLP(
                input_size=d_input, hidden_size=d_emb, output_size=d_emb, n_layers=0,
                activation=nn.Tanh, in_dropout=0.0, net_dropout=0.2)
        else: self.input_adapt = None
        # initialize the embedding table
        if word_emb is None:
            self.embedding = nn.Embedding(vocab_size, d_emb, padding_idx=self.padding_idx)
        else:
            embs = torch.tensor(word_emb, dtype=torch.float)
            assert (embs.shape[0], embs.shape[1]) == (vocab_size, d_emb) ,\
                    "shape of the pretrained embeddings do not match model specifications"
            self.embedding = nn.Embedding.from_pretrained(embs, freeze=False,
                                                          padding_idx=self.padding_idx)

    def initParams(self):
        for name, param in self.named_parameters():
            # do not initialize embeddings if pretrained values are used for init
            if "embedding" in name and self.word_emb is not None: continue
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            else:  # gain parameters of the layer norm
                nn.init.ones_(param)

    def numParameters(self):
        ''' Return the number of model parameters '''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def load(file):
        return torch.load(file)

    def save(self, file):
        file.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self, file)

class RnnDefmod(DefmodBase):
    '''
    RNN-based architecture for definition modeling.
    Uses a standard RNN, either LSTM or GRU, with optional gru-like activation as the last layer.
    '''

    def __init__(self, vocab_size, d_emb=256, d_input=256, d_hidden=256,
                 n_layers=2, base_arch='gru', use_gateact=True, in_dropout=0.05, net_dropout=0.2, maxlen=256,
                 word_emb=None, pad=vocab.PAD_ix, eos=vocab.EOS_ix, allvec=None, d_allvec=-1):
        '''
        :param n_layers: number of base RNN layers
        :param base_arch: base RNN architecture, 'gru' or 'lstm'
        :param use_gateact: if True, use gru-like gate activation as the last layer
                to combine the gloss emb. vector with a per-word rnn output
        :param allvec: if not None, gloss emb. vector will be a concat of several types of embeddings,
                all of size d_emb, and "sgns", ie word2vec must come first
                allvec can be either 'concat' or 'merge'
        :param d_allvecs: if allvec is True, the size of the concat. vector
        '''
        super(RnnDefmod, self).__init__(vocab_size=vocab_size, d_emb=d_emb, d_input=d_input,
                                        maxlen=maxlen, word_emb=word_emb, pad=pad, eos=eos,
                                        in_dropout=in_dropout, net_dropout=net_dropout)
        if base_arch == 'gru': RnnCls = GRU
        elif base_arch == 'lstm': RnnCls = LSTM
        else: raise ValueError(f'unknown rnn architecture: {base_arch}')
        self.d_hidden = d_hidden
        if allvec: # check the conditions
            assert allvec in ('merge', 'concat'), 'allvec must either be "concat" or "merge"'
            assert use_gateact, "No sense using allvec concatenation without gate activation"
            assert d_emb == d_input, "allvec not implemented in combination with input adapt (word emb != input gloss emb)"
            assert d_allvec >= d_emb, "allvec not implemented in case where d_allvec < d_emb"
        self.allvec = allvec
        self.d_allvec = d_allvec
        self._rnn = RnnCls(input_size=d_emb, hidden_size=d_hidden, num_layers=n_layers, dropout=0.0)
        self.base_arch = base_arch
        self.v_proj = nn.Linear(d_hidden, vocab_size)
        self.use_gateact = use_gateact
        if use_gateact: self._initGateActWeights()
        self.initParams()

    def _initGateActWeights(self):
        if self.allvec == 'concat': d_in = self.d_allvec
        else: d_in = self.d_emb
        if self.allvec == 'merge': self._Wmerg = nn.Linear(self.d_allvec, self.d_emb)
        #print(self.allvec, self.input_adapt, self.d_allvec, self.d_input, d_in)
        self._Wz = nn.Linear(d_in+self.d_hidden, self.d_hidden)
        self._Wr = nn.Linear(d_in+self.d_hidden, d_in)
        self._Wh = nn.Linear(d_in+self.d_hidden, self.d_hidden)
        self._sigmoid = nn.Sigmoid()
        self._tanh = nn.Tanh()

    def _applyGateActLayer(self, vectors, rnn_out):
        '''
        :param vectors: gloss. emb. vectors, either single gloss or multiple concat. glosses (allvec)
        :param rnn_out: outputs of the rnn
        '''
        # dropout
        vectors = self.in_drop(vectors)
        rnn_out = self.net_drop(rnn_out)
        # prepare data - cat vectors (pos. 0) to all sequence positions
        if self.allvec == 'merge': vectors = self._tanh(self._Wmerg(vectors))
        vectors = torch.unsqueeze(vectors, 0)
        vecstack = torch.cat([vectors] * len(rnn_out))
        rnn_cat = torch.cat((vecstack, rnn_out), -1)
        # apply gate activation network
        zt = self._sigmoid(self._Wz(rnn_cat))
        rt = self._sigmoid(self._Wr(rnn_cat))
        rnn_cat_vecweight = torch.cat((rt * vecstack, rnn_out), -1)
        rnn_out_tilde = self._tanh(self._Wh(rnn_cat_vecweight))
        rnn_out = (1.0 - zt) * rnn_out + zt * rnn_out_tilde
        return rnn_out

    def _applyWordClassifLayer(self, rnn_out):
        rnn_out = self.net_drop(rnn_out)
        word_prob = self.v_proj(rnn_out)
        return word_prob

    def forward(self, vector, input_sequence=None, predict_mode=False):
        # PUT DATA ON DEVICE
        device = next(self.parameters()).device
        if vector is not None: vector = vector.to(device)
        if input_sequence is not None: input_sequence = input_sequence.to(device)
        # PREPARE INPUT
        if not predict_mode: # normal forward call, for training
            embs = self.embedding(input_sequence)
            if self.input_adapt: vector = self.input_adapt(vector)
        else: # call by this.pred()
            # input_sequence contains both vectors (at seq. pos. 0) and already resolved embeddings
            vector = input_sequence[0] # squeezing, ie. removing first dim. is done automatically
            if not self.allvec: embs = input_sequence[1:]
            else: # allvec, embeddings are padded to vector (d_allvec) size, must reverse
                embs = input_sequence[1:, ..., :self.d_emb]
        # HANDLE 'ALLVEC' CASE
        # vector is a concatenation of several embedding, which are used for gate activation
        # however, only the first embeddings, (sgns or electra), will be used to 'seed' the RNN
        if self.allvec: input_vector = vector[..., :self.d_input]
        else: input_vector = vector
        # APPLY NETWORK
        # .. dropout
        input_vector = self.in_drop(input_vector)
        embs = self.in_drop(embs)
        # .. RNN
        seq = torch.cat([input_vector.unsqueeze(0), embs], dim=0)
        rnn_out, _ = self._rnn(seq)
        if self.use_gateact: rnn_out = self._applyGateActLayer(vector, rnn_out)
        return self._applyWordClassifLayer(rnn_out)

    @torch.no_grad()
    def pred_beamsearch(self, vector, decode_fn=None, beam_size=24, debug=False):
        # data
        # . current beam -> tensor of sequences (tensors of word indices)
        # .. tensor of logprobs of the beam sequences
        # from all the sequences in the beam -> create the batch of sequences
        # . padding is not needed -> all the lengths are the same
        # . vector first, embeddings follow
        # combine *all new candidates (continuations) with the current sequences:
        # . calculate logprobs of complete seqs: old seq + new continuations
        # . sort all the possible seqs by logbrop, take beam size top
        # . update sequences, update logprobs (add)
        # .
        device = next(self.parameters()).device
        vector = vector.to(device)
        if self.input_adapt: vector = self.input_adapt(vector)
        beam = None #torch.zeros(1, beam_size).to(device)
        logprobs = torch.zeros(beam_size).to(device)
        beam_elements = 1
        excluded_ixs = set([vocab.BOS_ix, self.padding_idx]) # valid for sentencepiece only!
        if debug:
            print(f'beam: {beam}')
            print(f'logprobs: {logprobs}')
        result = [] # a list of (sequence, logprob)
        for step_idx in range(self.maxlen):
            if step_idx == 0: # first step, only the (seed) vector is in play
                batch = vector.unsqueeze(0).unsqueeze(0)
                model_out = self(None, batch, True)
            else: # create batch form the vector (first pos) and beam sequences
                seqs = beam[:step_idx, :beam_elements]
                #seq_vecs = self.embedding(seqs)
                init_vecs = torch.cat([vector.unsqueeze(0)] * beam_elements)
                model_out = self(init_vecs, seqs)
                # vecstack = torch.cat([vectors] * len(rnn_out))
            #print(f'model_out\n: {model_out}')
            # ? exclude paddings v_dist[...,self.padding_idx] = -float("inf")
            cont_logprobs = F.log_softmax(model_out[-1], dim=-1)
            #print(f'cont_logprobs\n: {cont_logprobs}')
            new_sequences = []
            for b in range(beam_elements):
                for vi in range(self.vocab_size): # word indices
                    if vi == self.padding_idx or (step_idx > 0 and vi == vocab.BOS_ix): continue
                    #if vi in excluded_ixs: continue
                    new_sequences.append((b, vi, logprobs[b]+cont_logprobs[b, vi]))
            if debug: print(f'new_sequences len: {len(new_sequences)}')
            invlp = np.array([-(ns[2].cpu()) for ns in new_sequences])
            sort_ix = np.argsort(invlp)
            new_sequences = [ new_sequences[si] for si in sort_ix ]
            #new_sequences = sorted(new_sequences, key=lambda x: x[2], reverse=True)
            #new_sequences.sort(key=lambda x: x[2], reverse=True) # sort by logprob
            if beam_size < len(new_sequences): # take top beam size
                new_sequences = new_sequences[:beam_size]
            beam_elements = len(new_sequences)
            new_beam = torch.zeros(step_idx+2, beam_size, dtype=int).to(device)
            bi = 0
            for ns in new_sequences: # iter over new beam positions
                parent_ix, cont_ix, logprob = ns
                if cont_ix != self.eos_idx:
                    # copy previous sequences from the old beam
                    for wi in range(step_idx): new_beam[wi, bi] = beam[wi, parent_ix]
                    # add continuation word and
                    new_beam[step_idx, bi] = cont_ix
                    logprobs[bi] = logprob
                    bi += 1
                else:
                    #beam[step_idx, parent_ix] = self.eos_idx
                    res_seq = beam[:, parent_ix] #[ beam[wi, parent_ix] for wi in range(step_idx) ]
                    res_seq[step_idx] = self.eos_idx
                    #res_seq.append(self.eos_idx)
                    result.append((res_seq, logprob))
                    #print(f'result added: {decode_fn(res_seq)}')
                    beam_size -= 1
                    beam_elements -= 1
            if beam_size == 0: break
            beam = new_beam
            if debug:
                print(f'beam: {beam}')
                #print(f'logprobs: {logprobs}')
                print('sequences in the beam:')
                for i in range(beam_elements):
                    print(decode_fn(beam[:, i]))
                print()
        if len(result) == 0:
            beam[self.maxlen, 0] = self.eos_idx
            res_seq = beam[:, 0]  # [ beam[wi, parent_ix] for wi in range(step_idx) ]
            # res_seq.append(self.eos_idx)
            result.append((res_seq, -1))
        if debug:
            print('RESULT:')
            for re in result:
                seq, lp = re
                print(seq)
                print(decode_fn(seq), f'{lp}')
        return result[0][0]

    @torch.no_grad()
    def pred(self, vector, decode_fn=None, beam_size=64, verbose=False):
        # which device we should cast our variables to
        device = next(self.parameters()).device
        # how many examples are batched together
        batch_size = vector.size(0)
        vector = vector.to(device)
        # adapt input vectors to word embedding dimension, if necessary
        if self.input_adapt:
            vector = torch.stack([self.input_adapt(row) for row in vector])
        # Tensors will have this shape:
        # [Sequence, Batch, Beam, Continuation, *]
        # accumulation variable, keeping track of the best beams for each batched example
        generated_symbols = torch.zeros(0, batch_size, beam_size, dtype=torch.long).to(device)
        # which beams hold a completed sequence
        current_beam_size = 1
        has_stopped = torch.tensor([False] * (batch_size * current_beam_size)).to(device)
        # the input to kick-start the generation is the embedding, we start with the same input for each beam
        vector_src = vector.unsqueeze(1).expand(batch_size, current_beam_size, -1).reshape(1,  batch_size * current_beam_size, -1)
        src = vector_src
        #src_key_padding_mask = torch.tensor([[False] * (batch_size * current_beam_size)]).to(device)
        # variables needed to compute the score of each beam (geometric mean of probability of emission)
        logprobs = torch.zeros(batch_size, current_beam_size, dtype=torch.double).to(device)
        lengths = torch.zeros(batch_size * current_beam_size, dtype=torch.int).to(device)
        # generate tokens step by step
        for step_idx in range(self.maxlen):
            v_dist = self(None, src, True)[-1] # take only the last, most recently generated, char. position
            # don't generate padding tokens
            v_dist[...,self.padding_idx] = -float("inf")
            v_dist = F.log_softmax(v_dist, dim=-1)
            # for each beam, select the best candidate continuations
            new_logprobs, new_symbols = v_dist.topk(beam_size, dim=-1)
            # patch the output scores to zero-out items that have already stopped
            new_logprobs = new_logprobs.masked_fill(has_stopped.unsqueeze(-1), 0.0)
            # if the beam hasn't stopped, then it needs to produce at least an EOS
            # so we can just add one to beams that have not stopped to account for the current token
            lengths += (~has_stopped).int()

            # compute scores for each continuation
            ## recreate the score of the previous full sequence for all possible continuations
            logprobs_ = logprobs.view(batch_size * current_beam_size, 1).expand(batch_size * current_beam_size, beam_size)
            ## add the cost of each continuation
            logprobs_ = logprobs_ + new_logprobs
            ## average over the full sequence, ignoring padding items
            avg_logprobs = logprobs_ #/ lengths.unsqueeze(-1)
            ## select the `beam_size` best continuations overall, their matching scores will be `avg_logprobs`
            avg_logprobs, selected_beams = avg_logprobs.view(batch_size, current_beam_size * beam_size).topk(beam_size, dim=-1)
            ## select back the base score for the selected continuations
            logprobs = logprobs_.view(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(batch_size, beam_size)

            # add symbols of best continuations
            ## recreate the full previous sequence for all possible continuations
            generated_symbols_ = generated_symbols.view(-1, batch_size * current_beam_size, 1).expand(-1, batch_size * current_beam_size, beam_size)
            ## stack on the new symbols
            generated_symbols_ = torch.cat([generated_symbols_, new_symbols.unsqueeze(0)], dim=0)
            ## grab only the `beam_size` best continuations out of all possible continuations
            generated_symbols_ = generated_symbols_.view(-1, batch_size, current_beam_size * beam_size)
            generated_symbols = generated_symbols_.gather(-1, selected_beams.unsqueeze(0).expand(step_idx + 1, batch_size,  beam_size)).view(step_idx + 1, batch_size, beam_size)

            # recompute which beams have stopped, and what their lengths are
            ## reconstruct the lengths of all candidate continuations
            lengths = lengths.view(batch_size, current_beam_size, 1).expand(batch_size, current_beam_size, beam_size)
            ## retrieve the lengths of the selected beam continuations
            lengths = lengths.reshape(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(-1)
            ## reconstruct the halting state of all candidate continuations
            has_stopped = has_stopped.view(batch_size, current_beam_size, 1).expand(batch_size, current_beam_size, beam_size)
            ## retrieve the halting states of selected beam continuations
            has_stopped = has_stopped.reshape(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(-1)

            # flag which beams have terminated at the current step (i.e., whether they just produced an EOS)
            generated_symbols = generated_symbols.view(-1, batch_size * beam_size)
            generated_symbols[-1] = generated_symbols[-1].masked_fill(has_stopped, self.padding_idx)
            has_stopped = has_stopped | (generated_symbols.view(-1, batch_size * beam_size)[-1] == self.eos_idx).view(batch_size * beam_size)

            # TODO compute and set paddings ?
            # recompute padding mask on the basis of which continuations were selected
            #src_key_padding_mask = src_key_padding_mask.view(-1, batch_size, current_beam_size, 1).expand(-1, batch_size, current_beam_size, beam_size)
            #src_key_padding_mask = src_key_padding_mask.reshape(-1, batch_size, current_beam_size * beam_size)
            #src_key_padding_mask = src_key_padding_mask.gather(-1, selected_beams.unsqueeze(0).expand(step_idx + 1, batch_size,  beam_size)).view(step_idx + 1, batch_size * beam_size)
            #src_key_padding_mask = torch.cat([src_key_padding_mask, has_stopped.unsqueeze(0)], dim=0)

            # produce input for the next timestep
            orig_embs = self.embedding(generated_symbols)
            if self.allvec:
                embs = F.pad(input=orig_embs, pad=(0, self.d_allvec - self.d_emb),
                             mode='constant', value=0)
            else: embs = orig_embs
            src = torch.cat([vector_src.expand(1, beam_size, -1), embs], dim=0)
            # reshape to the familiar format
            generated_symbols = generated_symbols.view(-1, batch_size, beam_size)

            # if all beams have stopped, so do we
            if has_stopped.all():
                break
            # we update the number of sustained beam at the first iteration, since we know have `beam_size` candidates.
            current_beam_size = beam_size

        # select the most likely sequence for each batched item
        #print(f'generated_symbols:\n{generated_symbols}\n')
        max_scores, selected_beams = (logprobs / lengths.view(batch_size, beam_size)).topk(1, dim=1)
        output_sequence = generated_symbols.gather(1, selected_beams.unsqueeze(0).expand(step_idx + 1, batch_size, 1))
        if verbose: print(decode_fn(output_sequence.squeeze(-1)))
        return output_sequence.squeeze(-1)

# TODO (maybe later) extract common functionality of RevdictBase and Defmode in ModelBase class:
#  handling of common params: (pretrained embeddings, vocab, d_model, ...)
#  initialization of tensors
class TransformerDefmod(DefmodBase):
    """A transformer architecture for Definition Modeling."""

    def __init__(
        self, vocab_size, d_emb=256, d_input=256, n_head=4, n_layers=4, dropout=0.3, maxlen=256,
            word_emb=None, pad=vocab.PAD_ix, eos=vocab.EOS_ix
    ):
        super(TransformerDefmod, self).__init__(vocab_size=vocab_size, d_emb=d_emb, d_input=d_input,
                                        maxlen=maxlen, word_emb=word_emb, pad=pad, eos=eos)
        self.positional_encoding = PositionalEncoding(d_emb, dropout=dropout, max_len=maxlen)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_emb, nhead=n_head, dropout=dropout, dim_feedforward=d_emb * 2
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.v_proj = nn.Linear(d_emb, vocab_size)
        # initializing weights
        self.initParams()

    def generate_square_subsequent_mask(self, sz):
        "from Pytorch"
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, vector, input_sequence=None):
        device = next(self.parameters()).device
        embs = self.embedding(input_sequence)
        if self.input_adapt: vector = self.input_adapt(vector)
        seq = torch.cat([vector.unsqueeze(0), embs], dim=0)
        src = self.positional_encoding(seq)
        src_mask = self.generate_square_subsequent_mask(src.size(0)).to(device)
        src_key_padding_mask = torch.cat(
            [
                torch.tensor([[False] * input_sequence.size(1)]).to(device),
                (input_sequence == self.padding_idx),
            ],
            dim=0,
        ).t()
        transformer_output = self.transformer_encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        v_dist = self.v_proj(transformer_output)
        return v_dist

    @torch.no_grad()
    def pred(self, vector, decode_fn=None, beam_size=64, verbose=False):
        # which device we should cast our variables to
        device = next(self.parameters()).device
        vector = vector.to(device)
        # how many examples are batched together
        batch_size = vector.size(0)
        # adapt input vectors to word embedding dimension, if necessary
        if self.input_adapt:
            vector = torch.stack([self.input_adapt(row) for row in vector])

        # Tensors will have this shape:
        # [Sequence, Batch, Beam, Continuation, *]

        # accumulation variable, keeping track of the best beams for each batched example
        generated_symbols = torch.zeros(0, batch_size, beam_size, dtype=torch.long).to(device)

        # which beams hold a completed sequence
        current_beam_size = 1
        has_stopped = torch.tensor([False] * (batch_size * current_beam_size)).to(device)

        # the input to kick-start the generation is the embedding, we start with the same input for each beam
        vector_src = vector.unsqueeze(1).expand(batch_size, current_beam_size, -1).reshape(1,  batch_size * current_beam_size, -1)
        src = vector_src
        src_key_padding_mask = torch.tensor([[False] * (batch_size * current_beam_size)]).to(device)

        # variables needed to compute the score of each beam (geometric mean of probability of emission)
        logprobs = torch.zeros(batch_size, current_beam_size, dtype=torch.double).to(device)
        lengths = torch.zeros(batch_size * current_beam_size, dtype=torch.int).to(device)
        # generate tokens step by step
        for step_idx in range(self.maxlen):

            # generation mask
            src_mask = self.generate_square_subsequent_mask(src.size(0)).to(device)
            # positional encoding
            src_pe = self.positional_encoding(src)
            # transformer output
            transformer_output = self.transformer_encoder(
                src_pe, mask=src_mask, src_key_padding_mask=src_key_padding_mask.t()
            )[-1]
            # distribution over the full vocabulary
            v_dist = self.v_proj(transformer_output)
            # don't generate padding tokens
            v_dist[...,self.padding_idx] = -float("inf")
            v_dist = F.log_softmax(v_dist, dim=-1)

            # for each beam, select the best candidate continuations
            new_logprobs, new_symbols = v_dist.topk(beam_size, dim=-1)
            # patch the output scores to zero-out items that have already stopped
            new_logprobs = new_logprobs.masked_fill(has_stopped.unsqueeze(-1), 0.0)
            # if the beam hasn't stopped, then it needs to produce at least an EOS
            # so we can just add one to beams that have not stopped to account for the current token
            lengths += (~has_stopped).int()

            # compute scores for each continuation
            ## recreate the score of the previous full sequence for all possible continuations
            logprobs_ = logprobs.view(batch_size * current_beam_size, 1).expand(batch_size * current_beam_size, beam_size)
            ## add the cost of each continuation
            logprobs_ = logprobs_ + new_logprobs
            ## average over the full sequence, ignoring padding items
            avg_logprobs = logprobs_ #/ lengths.unsqueeze(-1)
            ## select the `beam_size` best continuations overall, their matching scores will be `avg_logprobs`
            avg_logprobs, selected_beams = avg_logprobs.view(batch_size, current_beam_size * beam_size).topk(beam_size, dim=-1)
            ## select back the base score for the selected continuations
            logprobs = logprobs_.view(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(batch_size, beam_size)

            # add symbols of best continuations
            ## recreate the full previous sequence for all possible continuations
            generated_symbols_ = generated_symbols.view(-1, batch_size * current_beam_size, 1).expand(-1, batch_size * current_beam_size, beam_size)
            ## stack on the new symbols
            generated_symbols_ = torch.cat([generated_symbols_, new_symbols.unsqueeze(0)], dim=0)
            ## grab only the `beam_size` best continuations out of all possible continuations
            generated_symbols_ = generated_symbols_.view(-1, batch_size, current_beam_size * beam_size)
            generated_symbols = generated_symbols_.gather(-1, selected_beams.unsqueeze(0).expand(step_idx + 1, batch_size,  beam_size)).view(step_idx + 1, batch_size, beam_size)

            # recompute which beams have stopped, and what their lengths are
            ## reconstruct the lengths of all candidate continuations
            lengths = lengths.view(batch_size, current_beam_size, 1).expand(batch_size, current_beam_size, beam_size)
            ## retrieve the lengths of the selected beam continuations
            lengths = lengths.reshape(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(-1)
            ## reconstruct the halting state of all candidate continuations
            has_stopped = has_stopped.view(batch_size, current_beam_size, 1).expand(batch_size, current_beam_size, beam_size)
            ## retrieve the halting states of selected beam continuations
            has_stopped = has_stopped.reshape(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(-1)

            # flag which beams have terminated at the current step (i.e., whether they just produced an EOS)
            generated_symbols = generated_symbols.view(-1, batch_size * beam_size)
            generated_symbols[-1] = generated_symbols[-1].masked_fill(has_stopped, self.padding_idx)
            has_stopped = has_stopped | (generated_symbols.view(-1, batch_size * beam_size)[-1] == self.eos_idx).view(batch_size * beam_size)

            # recompute padding mask on the basis of which continuations were selected
            src_key_padding_mask = src_key_padding_mask.view(-1, batch_size, current_beam_size, 1).expand(-1, batch_size, current_beam_size, beam_size)
            src_key_padding_mask = src_key_padding_mask.reshape(-1, batch_size, current_beam_size * beam_size)
            src_key_padding_mask = src_key_padding_mask.gather(-1, selected_beams.unsqueeze(0).expand(step_idx + 1, batch_size,  beam_size)).view(step_idx + 1, batch_size * beam_size)
            src_key_padding_mask = torch.cat([src_key_padding_mask, has_stopped.unsqueeze(0)], dim=0)

            # produce input for the next timestep
            src = torch.cat([vector_src.expand(1, beam_size, -1), self.embedding(generated_symbols)], dim=0)
            # reshape to the familiar format
            generated_symbols = generated_symbols.view(-1, batch_size, beam_size)

            # if all beams have stopped, so do we
            if has_stopped.all():
                break
            # we update the number of sustained beam at the first iteration, since we know have `beam_size` candidates.
            current_beam_size = beam_size

        # select the most likely sequence for each batched item
        # print(f'generated_symbols:\n{generated_symbols}\n')
        max_scores, selected_beams = (logprobs / lengths.view(batch_size, beam_size)).topk(1, dim=1)
        output_sequence = generated_symbols.gather(1, selected_beams.unsqueeze(0).expand(step_idx + 1, batch_size, 1))
        if verbose: print(decode_fn(output_sequence.squeeze(-1)))
        return output_sequence.squeeze(-1)


