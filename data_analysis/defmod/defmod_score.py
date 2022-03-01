'''
Script adapted from the original repo (https://github.com/TimotheeMickus/codwoe) file code/score.py
for the purpose of more flexible and fine grained analysis of defmod scores
'''

import argparse
import collections
import itertools
import json
import logging
import os
import pathlib
import sys
from copy import deepcopy

from data_analysis.data_utils import json_write

logger = logging.getLogger(pathlib.Path(__file__).name)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(handler)

os.environ["MOVERSCORE_MODEL"] = "distilbert-base-multilingual-cased"
import moverscore_v2 as mv_sc

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk import word_tokenize as tokenize

import numpy as np

import torch
import torch.nn.functional as F

import tqdm


def get_parser(parser=argparse.ArgumentParser(description="score a submission")):
    parser.add_argument(
        "submission_path",
        type=pathlib.Path,
        help="path to submission file to be scored, or to a directory of submissions to be scored",
    )
    parser.add_argument("--lang", type=str)
    parser.add_argument(
        "--submission_file",
        type=pathlib.Path
    )
    parser.add_argument(
        "--reference_file",
        type=pathlib.Path
    )
    parser.add_argument(
        "--reference_files_dir",
        type=pathlib.Path,
        help="directory containing all reference files",
        default=pathlib.Path("data"),
    )
    parser.add_argument(
        "--output_file",
        type=pathlib.Path,
        help="default path to print output",
        default=pathlib.Path("scores.txt"),
    )
    parser.add_argument(
        "--dset_output_file",
        type=pathlib.Path,
        help="path to print full test dataset enriched with scores and model-output glosses",
        default=pathlib.Path("output.json"),
    )
    return parser

def bleu(pred, target, smoothing_function=SmoothingFunction().method4):
    #print(f'[{pred}] [{target}]')
    return sentence_bleu([pred], target, smoothing_function=smoothing_function)


def mover_corpus_score(sys_stream, ref_streams, trace=0):
    """Adapted from the MoverScore github"""

    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]
    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]
    fhs = [sys_stream] + ref_streams
    corpus_score = 0
    pbar = tqdm.tqdm(desc="MvSc.", disable=None, total=len(sys_stream))
    all_scores = []
    for lines in itertools.zip_longest(*fhs):
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")
        hypo, *refs = lines
        idf_dict_hyp = collections.defaultdict(lambda: 1.0)
        idf_dict_ref = collections.defaultdict(lambda: 1.0)
        score = mv_sc.word_mover_score(
            refs,
            [hypo],
            idf_dict_ref,
            idf_dict_hyp,
            stop_words=[],
            n_gram=1,
            remove_subwords=False,
        )[0]
        pbar.update()
        corpus_score += score
        all_scores.append(score)
    pbar.close()
    corpus_score /= len(sys_stream)
    return corpus_score, all_scores


def eval_defmod(args):
    # 1. read contents
    ## define accumulators for lemma-level BLEU and MoverScore
    reference_lemma_groups = collections.defaultdict(list)
    all_preds, all_tgts = [], []
    ## reading data files
    with open(args.submission_file, "r") as fp:
        submission = sorted(json.load(fp), key=lambda r: r["id"])
    with open(args.reference_file, "r") as fp:
        reference = sorted(json.load(fp), key=lambda r: r["id"])
    # uncomment next line for quick testing
    # submission, reference = submission[:10], reference[:10]
    output = deepcopy(reference)
    # 2. compute scores
    ## compute sense-level BLEU
    assert len(submission) == len(reference), "Missing items in submission!"
    id_to_lemma = {}
    pbar = tqdm.tqdm(total=len(submission), desc="S-BLEU", disable=None)
    ix = 0
    for sub, ref in zip(submission, reference):
        assert sub["id"] == ref["id"], "Mismatch in submission and reference files!"
        out = output[ix]
        assert out["id"] == sub["id"]
        out["out-gloss"] = deepcopy(sub["gloss"])
        all_preds.append(sub["gloss"])
        all_tgts.append(ref["gloss"])
        sub["gloss"] = tokenize(sub["gloss"])
        ref["gloss"] = tokenize(ref["gloss"])
        sub["sense-BLEU"] = bleu(sub["gloss"], ref["gloss"])
        out["sense-BLEU"] = sub["sense-BLEU"]
        reference_lemma_groups[(ref["word"], ref["pos"])].append(ref["gloss"])
        id_to_lemma[sub["id"]] = (ref["word"], ref["pos"])
        ix += 1
        pbar.update()
    pbar.close()
    ## compute lemma-level BLEU
    id2out = {out["id"]:out for out in output}
    for sub in tqdm.tqdm(submission, desc="L-BLEU", disable=None):
        sub["lemma-BLEU"] = max(
            bleu(sub["gloss"], g)
            for g in reference_lemma_groups[id_to_lemma[sub["id"]]]
        )
        id = sub["id"]
        out = id2out[id]
        assert out["id"] == sub["id"]
        out["lemma-BLEU"] = sub["lemma-BLEU"]
    lemma_bleu_average = sum(s["lemma-BLEU"] for s in submission) / len(submission)
    sense_bleu_average = sum(s["sense-BLEU"] for s in submission) / len(submission)
    ## compute MoverScore
    # moverscore_average = np.mean(mv_sc.word_mover_score(
    #     all_tgts,
    #     all_preds,
    #     collections.defaultdict(lambda:1.),
    #     collections.defaultdict(lambda:1.),
    #     stop_words=[],
    #     n_gram=1,
    #     remove_subwords=False,
    #     batch_size=1,
    # ))
    moverscore_average, all_moverscores = mover_corpus_score(all_preds, [all_tgts])
    for ix, scr in enumerate(all_moverscores): output[ix]["moverscore"] = scr
    # 3. write results.
    # logger.debug(f"Submission {args.submission_file}, \n\tMvSc.: " + \
    #     f"{moverscore_average}\n\tL-BLEU: {lemma_bleu_average}\n\tS-BLEU: " + \
    #     f"{sense_bleu_average}"
    # )
    with open(args.output_file, "a") as ostr:
        print(f"MoverScore_{args.lang}:{moverscore_average}", file=ostr)
        print(f"BLEU_lemma_{args.lang}:{lemma_bleu_average}", file=ostr)
        print(f"BLEU_sense_{args.lang}:{sense_bleu_average}", file=ostr)
    json_write(args.dset_output_file, output)
    return (
        args.submission_file,
        moverscore_average,
        lemma_bleu_average,
        sense_bleu_average,
    )


def rank_cosine(preds, targets):
    assocs = F.normalize(preds) @ F.normalize(targets).T
    refs = torch.diagonal(assocs, 0).unsqueeze(1)
    ranks = (assocs >= refs).sum(1).float()
    assert ranks.numel() == preds.size(0)
    ranks = ranks.mean().item()
    return ranks / preds.size(0)


def eval_revdict(args, summary):
    # 1. read contents
    ## read data files
    with open(args.submission_file, "r") as fp:
        submission = sorted(json.load(fp), key=lambda r: r["id"])
    with open(args.reference_file, "r") as fp:
        reference = sorted(json.load(fp), key=lambda r: r["id"])
    vec_archs = sorted(
        set(submission[0].keys())
        - {
            "id",
            "gloss",
            "word",
            "pos",
            "concrete",
            "example",
            "f_rnk",
            "counts",
            "polysemous",
        }
    )
    ## define accumulators for rank-cosine
    all_preds = collections.defaultdict(list)
    all_refs = collections.defaultdict(list)

    assert len(submission) == len(reference), "Missing items in submission!"
    ## retrieve vectors
    for sub, ref in zip(submission, reference):
        assert sub["id"] == ref["id"], "Mismatch in submission and reference files!"
        for arch in vec_archs:
            all_preds[arch].append(sub[arch])
            all_refs[arch].append(ref[arch])

    torch.autograd.set_grad_enabled(False)
    all_preds = {arch: torch.tensor(all_preds[arch]) for arch in vec_archs}
    all_refs = {arch: torch.tensor(all_refs[arch]) for arch in vec_archs}

    # 2. compute scores
    MSE_scores = {
        arch: F.mse_loss(all_preds[arch], all_refs[arch]).item() for arch in vec_archs
    }
    cos_scores = {
        arch: F.cosine_similarity(all_preds[arch], all_refs[arch]).mean().item()
        for arch in vec_archs
    }
    rnk_scores = {
        arch: rank_cosine(all_preds[arch], all_refs[arch]) for arch in vec_archs
    }
    # 3. display results
    # logger.debug(f"Submission {args.submission_file}, \n\tMSE: " + \
    #     ", ".join(f"{a}={MSE_scores[a]}" for a in vec_archs) + \
    #     ", \n\tcosine: " + \
    #     ", ".join(f"{a}={cos_scores[a]}" for a in vec_archs) + \
    #     ", \n\tcosine ranks: " + \
    #     ", ".join(f"{a}={rnk_scores[a]}" for a in vec_archs) + \
    #     "."
    # )
    # all_archs = sorted(set(reference[0].keys()) - {"id", "gloss", "word", "pos"})
    with open(args.output_file, "a") as ostr:
        for arch in vec_archs:
            print(f"MSE_{summary.lang}_{arch}:{MSE_scores[arch]}", file=ostr)
            print(f"cos_{summary.lang}_{arch}:{cos_scores[arch]}", file=ostr)
            print(f"rnk_{summary.lang}_{arch}:{rnk_scores[arch]}", file=ostr)
    return (
        args.submission_file,
        *[MSE_scores.get(a, None) for a in vec_archs],
        *[cos_scores.get(a, None) for a in vec_archs],
    )

if __name__ == "__main__":
    eval_defmod(get_parser().parse_args())