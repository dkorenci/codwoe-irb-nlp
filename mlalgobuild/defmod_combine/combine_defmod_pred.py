import argparse, json, re
from glob import glob

from data_analysis.data_utils import json_load


def get_parser(
    parser=argparse.ArgumentParser(
        description="Run a embedding to embedding regression."
    ),
):
    parser.add_argument("--folder1", type=str)
    parser.add_argument("--folder2", type=str)
    parser.add_argument("--folderOut", type=str, default="~/")
    parser.add_argument("--idpart", type=int, default=2,
                        help="number of dot-separated strings at the start of filename that"
                             "uniquely indetify a file in both folders - used for file alignment")
    parser.add_argument("--format", type=str, choices=("dictV1", "lcase"),
                        help="the way the gloss predictions are formatted for final output")
    parser.add_argument("--strategy", type=str, choices=("fallback"), default="fallback",
                        help="strategy for combining two glosses for the same word.")
    return parser

# zamjeni, ako je jedna def.
#  prazna, kratka rijec (1-2 slova)
#  len <= 3 -> zamijeni ako je druga >= 3
#  ako nema alfabetskih znakova
# makni sve unutar ( ) - to prvo! - i točku na kraju
def combine_glosses(gls1, gls2, strategy):
    '''
    :return: is gls1 replaced (bool), resulting combination gloss
    '''
    #gls1 = gls1.strip()
    #gls2 = gls2.strip()
    # clean gls1 from (probably) non-informative content
    # glosses with everything within [] or () removed, including the enclosing parentheses
    gls1np = re.sub("[\(\[].*?[\)\]]", "", gls1).strip()
    # collapse ws
    gls1np = re.sub("\s+", " ", gls1np)
    #gls2np = re.sub("[\(\[].*?[\)\]]", "", gls2).strip()
    gls1alpha = any(c.isalpha() for c in gls1np)
    gls2alpha = any(c.isalpha() for c in gls2)
    out = None
    if strategy == 'fallback':
        if not gls2alpha: out = None # no alpha in gls2, do not replace
        elif not gls1alpha: out = gls2
        elif len(gls1np) <= 2: out = gls2
        #elif len(gls1np) <= 3 and len(gls2) > 3: out = gls2
    if out is not None: return True, gls2
    else: return False, gls1

def combine_files(f1, f2, args):
    '''
    Combine prediction from two json files.
    :param f1: primary predictions
    :param f2: fallback predictions
    :param args: cmdline args
    :return:
    '''
    jsn1, jsn2 = json_load(f1), json_load(f2)
    jsn2map = { itm['id']: itm['gloss'] for itm in jsn2 }
    jsn1.sort(key=lambda ji: ji["id"])
    result = []
    for itm in jsn1:
        id = itm['id']
        gls1 = itm['gloss']
        gls2 = jsn2map[id]
        is_replaced, glsC = combine_glosses(gls1, gls2, args.strategy)
        #print(f'{id}\n[{gls1}]\n[{gls2}]\n')
        if (is_replaced): print(f'{id}\n{gls1}\n{gls2}\n')
        result.append({'id':id, 'gloss':glsC})
    #print(jsn1[0])
    return result

def combine_predictions(args):
    '''
    Combine gloss predictions in files in two folders from two different models.
    Files must be matched by language and contain the same ids.
    :param args: cmdline args
    :return:
    '''
    for ff1 in glob(args.folder1+"*.json"):
        #print(ff1)
        idpart = '.'.join(ff1.split('/')[-1].split('.')[:args.idpart])
        #print(idpart)
        ff2 = glob(args.folder2+f'{idpart}*')[0]
        #print(ff2)
        res = combine_files(ff1, ff2, args)
        outf = args.folderOut
        with open(outf+idpart+'.json', 'w') as out_file:
            json.dump(res, out_file, indent=2)
    pass

if __name__ == '__main__':
    args = get_parser().parse_args()
    combine_predictions(args)