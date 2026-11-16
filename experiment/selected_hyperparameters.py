"""One generated-domain selection per method, reused unchanged across domains."""
import json
from pathlib import Path


def load_selection(path):
    return json.loads(Path(path).read_text(encoding='utf-8')) if path else {'methods':{}}


def selected_parameters(selection,method):
    key='ours_categorical_baa' if method=='ours' else method
    result=dict(selection.get('methods',{}).get(key,{}).get('parameters',{}))
    return result


def source_directory(root,method,seed,budget=500,*,selection=None,grid_layout=False):
    root=Path(root)
    if grid_layout:
        key='ours_categorical_baa' if method=='ours' else method
        identifier=selection['methods'][key]['configuration_id']
        root=root/identifier
    directory=root/method/f'seed-{seed}'
    return directory if budget is None else directory/'budgets'/f'episodes-{budget}'
