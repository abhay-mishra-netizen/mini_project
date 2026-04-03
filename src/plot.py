import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--x", type=str, default="round")
    parser.add_argument("--y", type=str, default="test_acc")
    parser.add_argument("--title", type=str, default="Training Curve")
    parser.add_argument("--save", type=str, default="outputs/plot.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    plt.figure(figsize=(8, 5))
    plt.plot(df[args.x], df[args.y])
    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.title(args.title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.save)
    print(f"Plot saved to: {args.save}")


if __name__ == "__main__":
    main()