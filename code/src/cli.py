import argparse

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PAD Evaluation with Fourier Features",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate and save visualization plots for cross-validation splits"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ALL",
        choices=["ALL", "PLUS", "IDIAP", "SCUT"],
        help="Select which dataset(s) to process. Options: ALL (default), PLUS, IDIAP, or SCUT"
    )

    parser.add_argument(
        "--print_corr",
        type=bool,
        default=False,
        help="Print the correlation between magnitude and phase features"
    )

    parser.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["1", "2", "both"],
        help="Select which task to run. Options: 1 (Task 1 only), 2 (Task 2 only), both (default - run both tasks)"
    )

    return parser.parse_args()
