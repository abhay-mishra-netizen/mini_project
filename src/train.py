import argparse
import csv
import os

import torch

from data import get_fl_setup, get_wwfl_setup
from fl_baseline import evaluate as fl_evaluate
from fl_baseline import fedavg
from models import get_model
from ww_fl import WWFLTrainer


def save_history(history, save_path):
    if not history:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def run_fl(args, device):
    client_loaders, test_loader = get_fl_setup(
        data_dir=args.data_dir,
        dataset=args.dataset,        # add this
        num_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        batch_size=args.fl_batch_size,  # changed
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    global_model = get_model(args.model)

    trained_model, history = fedavg(
        global_model=global_model,
        client_loaders=client_loaders,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.fl_lr,              # changed
        device=device,
    )

    test_loss, test_acc = fl_evaluate(
        trained_model, test_loader, device
    )
    print(f"\nFinal FL test loss: {test_loss:.4f}")
    print(f"Final FL test accuracy: {test_acc:.4f}")

    save_history(history, os.path.join(
        args.output_dir,
        f"fl_{args.dataset}_{args.model}_"
        f"{args.rounds}rounds_history.csv"  # descriptive name
    ))
    torch.save(
        trained_model.state_dict(),
        os.path.join(
            args.output_dir,
            f"fl_{args.dataset}_{args.model}_model.pt"
        )
    )


def run_wwfl(args, device):
    client_datasets, cluster_map, test_loader = \
        get_wwfl_setup(
            data_dir=args.data_dir,
            dataset=args.dataset,        # add this
            num_clients=args.num_clients,
            num_clusters=args.num_clusters,
            samples_per_client=args.samples_per_client,
            test_batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )

    trainer = WWFLTrainer(
        model_fn=lambda: get_model(args.model),
        client_datasets=client_datasets,
        cluster_map=cluster_map,
        test_loader=test_loader,
        num_rounds=args.rounds,
        clients_per_cluster_per_round=
            args.clients_per_cluster_per_round,
        local_epochs=args.local_epochs,
        batch_size=args.wwfl_batch_size,  # changed
        lr=args.wwfl_lr,                  # changed
        device=device,
        seed=args.seed,
        dataset_name=args.dataset         # add this
    )

    trained_model, history = trainer.fit()

    print(f"\nFinal WW-FL test loss: "
          f"{history[-1]['test_loss']:.4f}")
    print(f"Final WW-FL test accuracy: "
          f"{history[-1]['test_acc']:.4f}")

    save_history(history, os.path.join(
        args.output_dir,
        f"wwfl_{args.dataset}_{args.model}_"
        f"{args.rounds}rounds_history.csv"  # descriptive name
    ))
    torch.save(
        trained_model.state_dict(),
        os.path.join(
            args.output_dir,
            f"wwfl{args.dataset}_{args.model}_model.pt"
        )
    )

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str,
                        default="wwfl",
                        choices=["fl", "wwfl"])
    parser.add_argument("--model", type=str,
                        default="resnet9",
                        choices=["resnet9", "lenet"])

    # add dataset argument
    parser.add_argument("--dataset", type=str,
                        default="mnist",
                        choices=["mnist", "cifar10"])

    parser.add_argument("--data-dir", type=str,
                        default="./data")
    parser.add_argument("--output-dir", type=str,
                        default="./outputs")

    parser.add_argument("--rounds", type=int,
                        default=10)
    parser.add_argument("--local-epochs", type=int,
                        default=5)

    # separate batch sizes for FL and WW-FL
    parser.add_argument("--fl-batch-size", type=int,
                        default=8)
    parser.add_argument("--wwfl-batch-size", type=int,
                        default=80)

    parser.add_argument("--test-batch-size", type=int,
                        default=256)

    # separate learning rates
    parser.add_argument("--fl-lr", type=float,
                        default=0.005)
    parser.add_argument("--wwfl-lr", type=float,
                        default=0.05)

    parser.add_argument("--num-clients", type=int,
                        default=1000)
    parser.add_argument("--samples-per-client", type=int,
                        default=200)
    parser.add_argument("--num-clusters", type=int,
                        default=10)
    parser.add_argument("--clients-per-cluster-per-round",
                        type=int, default=10)
    parser.add_argument("--num-workers", type=int,
                        default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = args.device

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "fl":
        run_fl(args, device)
    else:
        run_wwfl(args, device)


if __name__ == "__main__":
    main()