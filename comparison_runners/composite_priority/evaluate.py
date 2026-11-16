"""Deploy one frozen weight vector on the existing generated test manifest."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from comparison_runners.evaluate_revision_nonlearning import execute,parse_args


if __name__=='__main__':
    args=parse_args()
    args.methods=['composite']
    execute(args)
