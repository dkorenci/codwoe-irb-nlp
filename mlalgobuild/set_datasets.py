import json
from pathlib import Path
from copy import deepcopy

raise NotImplementedError('This script is not implemented. Check and test code.')


# 1. Generate defmod dataset
dir_main = '../data'

languages = ["en", "es", "fr", "it", "ru"]
tasks = ["defmod", "revdict"]

# dsplit_srcs = ['train']
# files_src = [ '{lang}.{dsplit}.json' ]

dsplit_srcs = ['dev']
files_src = [ '{lang}.{dsplit}.json' ]

dsplit_dests = [ 'dev' ]
files_dest = [ '{lang}.{dsplit}.{task}.json' ]
suffix_dest = '{lang}.{dsplit}.{task}.{idx}'

for src_id, dsplit_src in enumerate(dsplit_srcs):
    dir_src = dsplit_src
    file_src = files_src[src_id]

    # for d in dirs_dest:
    #     (Path(dir_main) / dir_dest+"-"+dsplit_src).mkdir(parents=True, exist_ok=True)

    for dest_id, dsplit_dest in enumerate(dsplit_dests):
        dir_dest = (Path(dir_main) / "{}".format(dsplit_dest, dsplit_src))
        dir_dest.mkdir(parents=True, exist_ok=True)
        
        for lang in languages:
            for task in tasks:
                info_src=dict(lang=lang, dsplit=dsplit_src, task=task)
                info_dest=dict(lang=lang, dsplit=dsplit_dest, task=task)

                src = Path(dir_main) / dir_src / file_src.format(**info_src)
                dest = dir_dest / files_dest[dest_id].format(**info_dest)
                print(src)
                print(dest)

                data = []
                idx = 0

                # Load and edit json file
                with open(src, "r") as f:
                    items = json.load(f)

                    for i, item in enumerate(items):
                        # suffix_src = '.'.join(item['id'].split('.')[:-1])
                        word_list = []
                        for word in item['gloss'].split(' '):
                            word_list.append(word)
                            idx += 1
                            new_item = deepcopy(item)
                            new_item['gloss'] = " ".join(word_list)
                            new_item['id'] = suffix_dest.format(**info_dest, idx=idx)
                            # print(new_item['id'], new_item['gloss'])
                            data.append(item)
                            
                # Save json file to destination
                with open(dest, 'w') as f:
                    json.dump(data, f)import json




# 2. Generate data sim
dir_main = '../data'

languages = ["en", "es", "fr", "it", "ru"]
tasks = ["defmod", "revdict"]

dsplit_srcs = ['dev', 'trial']
files_src = [ '{lang}.{dsplit}.json', '{lang}.{dsplit}.complete.json' ]

dirs_dest = [ 'test', 'reference' ]
files_dest = [ '{lang}.{dsplit}.{task}.json', '{lang}.{dsplit}.{task}.complete.json' ]
dsplit_dest = 'test'
suffix_dest = '{lang}.{dsplit}.{task}'

for src_id, dsplit_src in enumerate(dsplit_srcs):
    dir_src = dsplit_src
    file_src = files_src[src_id]

    # for d in dirs_dest:
    #     (Path(dir_main) / dir_dest+"-"+dsplit_src).mkdir(parents=True, exist_ok=True)

    for dest_id, dir_dest in enumerate(dirs_dest):
        dir_dest = (Path(dir_main) / "{}-{}".format(dir_dest, dsplit_src))
        dir_dest.mkdir(parents=True, exist_ok=True)
        
        for lang in languages:
            for task in tasks:            
                info_src=dict(lang=lang, dsplit=dsplit_src, task=task)
                info_dest=dict(lang=lang, dsplit=dsplit_dest, task=task)

                src = Path(dir_main) / dir_src / file_src.format(**info_src)
                dest = dir_dest / files_dest[dest_id].format(**info_dest)
                print(dest)

                data = []

                # Load and edit json file
                with open(src, "r") as f:
                    items = json.load(f)

                    for i, item in enumerate(items):
                        suffix_src = '.'.join(item['id'].split('.')[:-1])
                        item['id'] = item['id'].replace(suffix_src, suffix_dest.format(**info_dest))
                        data.append(item)

                # Save json file to destination
                with open(dest, 'w') as f:
                    json.dump(data, f)
