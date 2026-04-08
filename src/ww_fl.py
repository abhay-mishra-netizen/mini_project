import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def train_cluster_model(
    model: nn.Module,
    dataloader,
    epochs: int,
    lr: float = 0.05,
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], float, float, int]:
    model = model.to(device)
    model.train()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for _ in range(epochs):
        for data, target in dataloader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (output.argmax(dim=1) == target).sum().item()
            total_seen += batch_size

    avg_loss = total_loss / total_seen if total_seen > 0 else 0.0
    avg_acc = total_correct / total_seen if total_seen > 0 else 0.0

    return copy.deepcopy(model.state_dict()), avg_loss, avg_acc, total_seen


def aggregate_cluster_models(
    weights_list: List[Dict[str, torch.Tensor]],
    data_sizes: List[int],
) -> Dict[str, torch.Tensor]:
    if not weights_list:
        raise ValueError("weights_list cannot be empty")

    total_data = sum(data_sizes)
    if total_data == 0:
        raise ValueError("Total data size cannot be zero")

    aggregated = copy.deepcopy(weights_list[0])

    for key in aggregated.keys():
        aggregated[key] = weights_list[0][key] * (data_sizes[0] / total_data)
        for idx in range(1, len(weights_list)):
            aggregated[key] += weights_list[idx][key] * (data_sizes[idx] / total_data)

    return aggregated


def evaluate(model: nn.Module, dataloader, device: str = "cpu") -> Tuple[float, float]:
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(data)
            loss = criterion(output, target)

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (output.argmax(dim=1) == target).sum().item()
            total_seen += batch_size

    avg_loss = total_loss / total_seen if total_seen > 0 else 0.0
    avg_acc = total_correct / total_seen if total_seen > 0 else 0.0
    return avg_loss, avg_acc


def wwfl_train(
    global_model: nn.Module,
    data_manager,
    test_loader,
    num_rounds: int,
    local_epochs: int,
    lr: float,
    device: str,
    on_round_end=None,
):
    global_model = global_model.to(device)
    round_history = []
    cluster_history = []

    for round_idx in range(1, num_rounds + 1):
        print(f"[WWFL] Round {round_idx}/{num_rounds}")
        round_clusters = data_manager.prepare_round_cluster_loaders()
        current_round_cluster_rows = []

        cluster_weights = []
        cluster_sizes = []
        weighted_losses = []
        weighted_accs = []

        for cluster_id in sorted(round_clusters.keys()):
            cluster_info = round_clusters[cluster_id]
            cluster_model = copy.deepcopy(global_model)

            weights, train_loss, train_acc, num_samples = train_cluster_model(
                model=cluster_model,
                dataloader=cluster_info["loader"],
                epochs=local_epochs,
                lr=lr,
                device=device,
            )

            cluster_weights.append(weights)
            cluster_sizes.append(num_samples)
            weighted_losses.append(train_loss * num_samples)
            weighted_accs.append(train_acc * num_samples)

            cluster_row = {
                "round": round_idx,
                "cluster_id": cluster_id,
                "sampled_clients": len(cluster_info["sampled_client_ids"]),
                "new_clients_added": len(cluster_info["new_client_ids"]),
                "cluster_dataset_size": num_samples,
                "train_loss": train_loss,
                "train_acc": train_acc,
            }
            current_round_cluster_rows.append(cluster_row)

            print(
                f"  cluster={cluster_id:02d} "
                f"sampled={len(cluster_info['sampled_client_ids'])} "
                f"new={len(cluster_info['new_client_ids'])} "
                f"size={num_samples} "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f}"
            )

        aggregated_weights = aggregate_cluster_models(cluster_weights, cluster_sizes)
        global_model.load_state_dict(aggregated_weights)

        total_samples = sum(cluster_sizes)
        cluster_train_loss = sum(weighted_losses) / total_samples if total_samples > 0 else 0.0
        cluster_train_acc = sum(weighted_accs) / total_samples if total_samples > 0 else 0.0
        test_loss, test_acc = evaluate(global_model, test_loader, device=device)

        round_row = {
            "round": round_idx,
            "cluster_train_loss": cluster_train_loss,
            "cluster_train_acc": cluster_train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        round_history.append(round_row)
        for cluster_row in current_round_cluster_rows:
            cluster_row["test_loss"] = test_loss
            cluster_row["test_acc"] = test_acc
        cluster_history.extend(current_round_cluster_rows)

        print(
            f"  summary round={round_idx} "
            f"cluster_train_loss={cluster_train_loss:.4f} "
            f"cluster_train_acc={cluster_train_acc:.4f} "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.4f}"
        )
        if on_round_end is not None:
            on_round_end(global_model, round_history, cluster_history)

    return global_model, round_history, cluster_history

