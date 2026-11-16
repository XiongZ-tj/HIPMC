"""Command-line entry point for scheduling experiments."""

from comparison_runners.revision_workflow.run import execute, parse_args


if __name__ == "__main__":
    options, forwarded = parse_args()
    execute(options, forwarded)
