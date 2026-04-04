import argparse
import csv
import os

import torch

from data import get_fl_setup, get_wwfl_setup
from fl_baseline import evaluate as fl_evaluate
from fl_baseline import fedavg
from models import get_model
from ww_fl import WWFLTrainer
from ww_fl_crypten import WWFLCrypTenTrainer


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
        num_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        batch_size=args.batch_size,
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
        lr=args.lr,
        device=device,
    )

    test_loss, test_acc = fl_evaluate(trained_model, test_loader, device)
    print(f"\nFinal FL test loss: {test_loss:.4f}")
    print(f"Final FL test accuracy: {test_acc:.4f}")

    save_history(history, os.path.join(args.output_dir, "fl_history.csv"))
    torch.save(trained_model.state_dict(), os.path.join(args.output_dir, "fl_model.pt"))


def run_wwfl(args, device):
    client_datasets, cluster_map, test_loader = get_wwfl_setup(
        data_dir=args.data_dir,
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
        clients_per_cluster_per_round=args.clients_per_cluster_per_round,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )

    trained_model, history = trainer.fit()

    print(f"\nFinal WW-FL test loss: {history[-1]['test_loss']:.4f}")
    print(f"Final WW-FL test accuracy: {history[-1]['test_acc']:.4f}")

    save_history(history, os.path.join(args.output_dir, "wwfl_history.csv"))
    torch.save(trained_model.state_dict(), os.path.join(args.output_dir, "wwfl_model.pt"))


def run_wwfl_crypten(args, device):
    client_datasets, cluster_map, test_loader = get_wwfl_setup(
        data_dir=args.data_dir,
        num_clients=args.num_clients,
        num_clusters=args.num_clusters,
        samples_per_client=args.samples_per_client,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    trainer = WWFLCrypTenTrainer(
        model_fn=lambda: get_model(args.model),
        client_datasets=client_datasets,
        cluster_map=cluster_map,
        test_loader=test_loader,
        num_rounds=args.rounds,
        clients_per_cluster_per_round=args.clients_per_cluster_per_round,
        local_epochs=args.local_epochs,
        lr=args.lr,
        device=device,
        seed=args.seed,
        keep_cluster_data_across_rounds=args.keep_cluster_data_across_rounds,
    )

    trained_model, history = trainer.fit()

    print(f"\nFinal CrypTen WW-FL test loss: {history[-1]['test_loss']:.4f}")
    print(f"Final CrypTen WW-FL test accuracy: {history[-1]['test_acc']:.4f}")

    save_history(history, os.path.join(args.output_dir, "wwfl_crypten_history.csv"))
    torch.save(
        trained_model.state_dict(),
        os.path.join(args.output_dir, "wwfl_crypten_model.pt"),
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="wwfl",
        choices=["fl", "wwfl", "wwfl_crypten"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet9",
        choices=["resnet9", "lenet"],
    )

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")

    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.05)

    parser.add_argument("--num-clients", type=int, default=1000)
    parser.add_argument("--samples-per-client", type=int, default=200)
    parser.add_argument("--num-clusters", type=int, default=10)
    parser.add_argument("--clients-per-cluster-per-round", type=int, default=10)

    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--keep-cluster-data-across-rounds",
        action="store_true",
        help="Only relevant for wwfl_crypten. If used, each cluster keeps accumulating old selected client data.",
    )

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
    elif args.mode == "wwfl":
        run_wwfl(args, device)
    else:
        run_wwfl_crypten(args, device)


if __name__ == "__main__":
    main()
