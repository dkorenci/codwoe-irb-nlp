'''
Dataset is a loader of raw json data and transformer of raw data to tensors.
'''

import json, torch, random
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
from collections import defaultdict

from lang_resources.sentencepice_build import createLoadSpm
from lang_resources.vocab import SpmVocab, PAD_ix

SUPPORTED_ARCHS = ("sgns", "char")
DATASET_EMB_SIZE = 256

# A dataset is a container object for the actual data
class JSONDataset2(Dataset):
    ''' Load codwoe json dataset and tranform it to tensors '''

    def __init__(self, file, vocab, maxlen=256, lowercase=False):
        '''
        :param file: path of the json file
        :param vocab: mapping of strings to integers, Vocab instance
        :param maxlen: maximum tensorize gloss length
        :param lowercase: lowercase orig. glosses (before tokenizing)
        '''
        self._vocab = vocab
        with open(file, "r") as istr:
            self.items = json.load(istr)
        # preparse data
        for json_dict in self.items:
            if "gloss" in json_dict: # in definition modeling test datasets, gloss targets are absent
                if lowercase: json_dict['gloss'] = json_dict['gloss'].lower()
                json_dict['gloss_tensor'] = torch.tensor(self._vocab.encode(json_dict['gloss']))
                if maxlen: json_dict["gloss_tensor"] = json_dict["gloss_tensor"][:maxlen]
            for arch in SUPPORTED_ARCHS: # in reverse dictionary test datasets, vector targets are absent
                if arch in json_dict:
                    json_dict[f"{arch}_tensor"] = torch.tensor(json_dict[arch])
            if "electra" in json_dict:
                json_dict["electra_tensor"] = torch.tensor(json_dict["electra"])
        self.has_gloss = "gloss" in self.items[0]
        self.has_vecs = SUPPORTED_ARCHS[0] in self.items[0]
        self.has_electra = "electra" in self.items[0]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def maxVocabIndex(self):
        return self._vocab.maxIndex()

    # we're adding this method to simplify the code in our predictions of
    # glosses
    @torch.no_grad()
    def decode(self, tensor):
        """Convert a sequence of indices (possibly batched) to tokens"""
        if tensor.dim() == 2:
            # we have batched tensors of shape [Seq x Batch]
            decoded = []
            for tensor_ in tensor.t():
                decoded.append(self.decode(tensor_))
            return decoded
        else:
            return self._vocab.decode(tensor.tolist())

    def save(self, file):
        torch.save(self, file)

    @staticmethod
    def load(file):
        return torch.load(file)


# A sampler allows you to define how to select items from your Dataset. Torch
# provides a number of default Sampler classes
class TokenSampler(Sampler):
    """Produce batches with up to `batch_size` tokens in each batch"""

    def __init__(
            self, dataset, batch_size=150, size_fn=len, drop_last=False, shuffle=True
    ):
        """
        args: `dataset` a torch.utils.data.Dataset (iterable style)
              `batch_size` the maximum number of tokens in a batch
              `size_fn` a callable that yields the number of tokens in a dataset item
              `drop_last` if True and the data can't be divided in exactly the right number of batch, drop the last batch
              `shuffle` if True, shuffle between every iteration
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.size_fn = size_fn
        self._len = None
        self.drop_last = drop_last
        self.shuffle = True

    def __iter__(self):
        indices = range(len(self.dataset))
        if self.shuffle:
            indices = list(indices)
            random.shuffle(indices)
        i = 0
        selected = []
        numel = 0
        longest_len = 0
        for i in indices:
            if numel + self.size_fn(self.dataset[i]) > self.batch_size:
                if selected:
                    yield selected
                selected = []
                numel = 0
            numel += self.size_fn(self.dataset[i])
            selected.append(i)
        if selected and not self.drop_last:
            yield selected

    def __len__(self):
        if self._len is None:
            self._len = round(
                sum(self.size_fn(self.dataset[i]) for i in range(len(self.dataset)))
                / self.batch_size
            )
        return self._len


# DataLoaders give access to an iterator over the dataset, using a sampling
# strategy as defined through a Sampler.
def get_dataloader(dataset, batch_size=1024, shuffle=True, allvec=False):
    """produce dataloader.
    args: `dataset` a torch.utils.data.Dataset (iterable style)
          `batch_size` the maximum number of tokens in a batch
          `shuffle` if True, shuffle between every iteration
    :return: vec_size (final size of the embeddings vector in the dataset), dataloader
    """
    # some constants for the closures
    has_gloss = dataset.has_gloss
    has_vecs = dataset.has_vecs
    has_electra = dataset.has_electra

    if allvec:
        vec_size = 0
        if has_vecs: vec_size += 2*DATASET_EMB_SIZE
        if has_electra: vec_size += DATASET_EMB_SIZE
    else: vec_size = DATASET_EMB_SIZE

    # the collate function has to convert a list of dataset items into a batch
    def do_collate(json_dicts):
        """collates example into a dict batch; produces ands pads tensors"""
        batch = defaultdict(list)
        for jdict in json_dicts:
            for key in jdict:
                batch[key].append(jdict[key])
        if has_gloss:
            batch["gloss_tensor"] = pad_sequence(
                batch["gloss_tensor"], padding_value=PAD_ix, batch_first=False
            )
        if not allvec:
            if has_vecs:
                for arch in SUPPORTED_ARCHS:
                    batch[f"{arch}_tensor"] = torch.stack(batch[f"{arch}_tensor"])
            if has_electra:
                batch["electra_tensor"] = torch.stack(batch["electra_tensor"])
        else: # crate one tensor with all vectors concatenated
            sgns = torch.stack(batch[f"sgns_tensor"]) # MUST BE FIRST!
            char = torch.stack(batch[f"char_tensor"])
            vecs = [sgns, char]
            if has_electra:
                electra = torch.stack(batch[f"electra_tensor"])
                # TODO: solve through params, this is a quick-patch to seed the RNN with
                #  electra vectors, by putting them in the first position
                # vecs.append(electra)
                vecs.insert(0, electra)
            batch["allvec_tensor"] = torch.cat(vecs, dim=1)
        return dict(batch)

    if dataset.has_gloss:
        # we try to keep the amount of gloss tokens roughly constant across all
        # batches.
        def do_size_item(item):
            """retrieve tensor size, so as to batch items per elements"""
            return item["gloss_tensor"].numel()

        return vec_size, DataLoader(
            dataset,
            collate_fn=do_collate,
            batch_sampler=TokenSampler(
                dataset, batch_size=batch_size, size_fn=do_size_item, shuffle=shuffle
            ),
        )
    else:
        # there's no gloss, hence no gloss tokens, so we use a default batching
        # strategy.
        return vec_size, DataLoader(
            dataset, collate_fn=do_collate, batch_size=batch_size, shuffle=shuffle
        )

def check_dataset(dataset, keys):
    """Check and stop if dataset contains no data with given keys."""
    if "gloss" in keys:
        assert dataset.has_gloss, "Dataset contains no gloss."
    if "electra" in keys:
        assert dataset.has_electra, "Datatset contains no electra."
    if "sgns" in keys or "char" in keys:
        assert dataset.has_vecs, "Datatset contains no vector."
    if "allvec" in keys:
        assert dataset.has_vecs, "Datatset contains no vector."
    return True

def construct_dataloader(subset, data_file, vocab_lang, vocab_subset, vocab_subdir, vocab_type, vocab_size,
                   input_key, output_key, batch_size=1024, shuffle=False, maxlen=256, lowercase=False):
    '''
    :return: vec_size (final size of the embeddings vector in the dataset), dataloader
    '''
    # create vocab and dataset
    if vocab_type == 'sentencepiece':
        vocab = createLoadSpm(vocab_lang, vocab_subset, vocab_subdir, dict_size=vocab_size)
        vocab = SpmVocab(vocab)
    else: raise ValueError(f'unsupported vocab type: {vocab_type}')
    dataset = JSONDataset2(data_file, vocab, maxlen=maxlen, lowercase=lowercase)
    # chack
    if subset in ['train', 'dev']: check_dataset(dataset, [input_key, output_key])
    elif subset == 'test': check_dataset(dataset, [input_key])
    else: raise ValueError(f'unrecognized subset: {subset}')
    return get_dataloader(dataset, shuffle=shuffle, batch_size=batch_size, allvec=(input_key=='allvec'))
