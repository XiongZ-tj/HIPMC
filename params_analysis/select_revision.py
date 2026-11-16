"Record author-selected configurations from generated-domain evidence."
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment.artifacts import atomic_json_save
from experiment.selected_hyperparameters import load_selection
from experiment.revision_protocol import LEARNED_METHODS,TRAINING_SEEDS
from params_analysis.grid import select_configurations


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=LEARNED_METHODS,required=True)
    p.add_argument('--configuration-id',required=True)
    p.add_argument('--reason',required=True)
    p.add_argument('--output',type=Path,default=ROOT/'experiment_manifests/selected_generated_hyperparameters.json')
    args=p.parse_args()
    cfg=select_configurations([args.configuration_id])[0]
    artifact=load_selection(args.output) if args.output.exists() else dict(methods={})
    artifact.update(selection_domain='generated',selection_evidence='training_stability_and_common_validation',
                    test_results_used_for_selection=False,training_seeds=list(TRAINING_SEEDS))
    artifact['methods'][args.method]=dict(configuration_id=cfg.configuration_id,
        parameters=dict(learning_rate=cfg.learning_rate,clip_ratio=cfg.clip_ratio,gamma=cfg.discount_factor),
        rationale=args.reason)
    atomic_json_save(artifact,args.output)
    print(f'[PARAM-SELECTION] {args.method} {cfg.configuration_id} -> {args.output}',flush=True)


if __name__=='__main__':main()
