import pandas as pd
from pathlib import Path

DMOD_METRICS = ['MvSc.', 'S-BLEU', 'L-BLEU']
LANGS = ['EN', 'ES', 'FR', 'IT', 'RU']

ORIG_DEFMOD_RES = '/data/code/semeval2022/codwoe-solve/codwoe_git/rankings/submission_scores/res_defmod.csv'
ORID_REVDICT_RES = [
    '/data/code/semeval2022/codwoe-solve/codwoe_git/rankings/submission_scores/res_revdict-sgns.csv',
    '/data/code/semeval2022/codwoe-solve/codwoe_git/rankings/submission_scores/res_revdict-electra.csv',
    '/data/code/semeval2022/codwoe-solve/codwoe_git/rankings/submission_scores/res_revdict-char.csv',
]

def orig_rankings(resfile):
    ''' Orig. defmod ranking code from the official competition repo '''
    df = pd.read_csv(resfile)

    def get_sorted_vals(colname):
        return sorted(df[colname].dropna(), reverse=True)

    for colname in [f"{lang} {metric}" for lang in LANGS for metric in DMOD_METRICS]:
        sorted_vals = get_sorted_vals(colname)

        def float_to_rank(cell):
            if pd.isna(cell): return cell
            return sum(i >= cell for i in sorted_vals)

        df[colname] = df[colname].apply(float_to_rank)
    df.to_csv('results_defmod.csv', index=False)
    df_ranks = df.groupby('user').min()
    for lang in LANGS:
        def get_mean_rank(row):
            metrics = [row[f"{lang} {metric}"] for metric in DMOD_METRICS]
            if any(map(pd.isna, metrics)): return pd.NA
            return sum(metrics) / len(metrics)

        df_ranks[f"Rank {lang}"] = df_ranks.apply(get_mean_rank, axis=1)
    #del df_ranks['Date']
    #del df_ranks['filename']
    df_ranks.to_csv('defmod_rankings-per-users.csv')

def negate_revdict_metrics(df):
    ''' Negate columns, in place, of revdict metrics that need to be minimized. '''
    neg = lambda x: -x
    for cname, _ in df.iteritems():
        if 'mse' in cname.lower() or 'rank' in cname.lower():
            df[cname] = df[cname].apply(neg)
    return df

def best_scores_peruser(orig_res_file, outfile, revdict=False):
    ''' Create a table with top scores for each user. '''
    df_orig = pd.read_csv(orig_res_file)
    users = df_orig['user']
    userset = set(u for u in users)
    bestdf = pd.DataFrame()
    if revdict: df_orig = negate_revdict_metrics(df_orig)
    for u in userset:
        udf = df_orig[df_orig['user'] == u]
        umax = udf.max(axis=0)
        print(umax)
        bestdf = bestdf.append(umax, ignore_index=True)
    #print(bestdf.columns)
    bestdf.drop(columns=['Comments', 'Date', 'filename'], inplace=True)
    if revdict: bestdf = negate_revdict_metrics(bestdf)
    #print(bestdf)
    if not outfile: outfile = 'bestdf.csv'
    bestdf.to_csv(outfile)

def scores_peruser(orig_res_file, user='dkorenci', outfile=None):
    ''' Extract user-specific score data. '''
    df_orig = pd.read_csv(orig_res_file)
    udf = df_orig[df_orig['user'] == user].copy()
    udf.drop(columns=['Comments', 'Date', 'user'], inplace=True)
    if not outfile: outfile = f'{user}_scores.csv'
    udf.to_csv(outfile)

def revdict_scores_peruser():
    for file in ORID_REVDICT_RES:
        scores_peruser(file, outfile=Path(file).with_suffix('.irbnlp.csv'))

def revdict_best_scores_peruser():
    for file in ORID_REVDICT_RES:
        best_scores_peruser(file, revdict=True,
                            outfile=Path(file).with_suffix('.bestperuser.csv'))

if __name__ == '__main__':
    #best_scores_peruser(ORIG_DEFMOD_RES)
    #scores_peruser(ORIG_DEFMOD_RES)
    #revdict_scores_peruser()
    revdict_best_scores_peruser()