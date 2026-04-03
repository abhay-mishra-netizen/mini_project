import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader


class WWFLTrainer:
    def __init__(
        self,
        model_fn,
        client_datasets,
        cluster_map,
        test_loader,
        num_rounds=500,
        clients_per_cluster_per_round=10,
        local_epochs=5,
        batch_size=80,
        lr=0.05,
        device="cuda",
        seed=42,
    ):
        self.model_fn = model_fn
        self.client_datasets = client_datasets
        self.cluster_map = cluster_map
        self.test_loader = test_loader
        self.num_rounds = num_rounds
        self.clients_per_cluster_per_round = clients_per_cluster_per_round
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed

        self.global_model = self.model_fn().to(device)
        self.cluster_memory = {cluster_id: [] for cluster_id in self.cluster_map}
        self.history = []

    def _sample_clients(self, cluster_id, round_idx):
        rng = random.Random(self.seed + 1000 * round_idx + cluster_id)
        client_ids = self.cluster_map[cluster_id]
        k = min(self.clients_per_cluster_per_round, len(client_ids))
        return rng.sample(client_ids, k)

    def _get_cluster_dataset(self, cluster_id, selected_client_ids):
        new_parts = [self.client_datasets[client_id] for client_id in selected_client_ids]
        self.cluster_memory[cluster_id].extend(new_parts)

        if len(self.cluster_memory[cluster_id]) == 1:
            return self.cluster_memory[cluster_id][0]

        return ConcatDataset(self.cluster_memory[cluster_id])

    def _train_cluster_model(self, cluster_dataset):
        model = self.model_fn().to(self.device)
        model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))
        model.train()

        loader = DataLoader(
            cluster_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for _ in range(self.local_epochs):
            for data, target in loader:
                data = data.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * target.size(0)
                total_correct += (output.argmax(dim=1) == target).sum().item()
                total_seen += target.size(0)

        avg_loss = total_loss / total_seen if total_seen > 0 else 0.0
        avg_acc = total_correct / total_seen if total_seen > 0 else 0.0

        return copy.deepcopy(model.state_dict()), len(cluster_dataset), avg_loss, avg_acc

    def _aggregate_cluster_models(self, cluster_states, cluster_sizes):
        new_state = copy.deepcopy(cluster_states[0])
        total_size = sum(cluster_sizes)

        for key in new_state.keys():
            new_state[key] = cluster_states[0][key] * (cluster_sizes[0] / total_size)
            for i in range(1, len(cluster_states)):
                new_state[key] += cluster_states[i][key] * (cluster_sizes[i] / total_size)

        self.global_model.load_state_dict(new_state)

    def evaluate(self):
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.global_model(data)
                loss = criterion(output, target)

                total_loss += loss.item() * target.size(0)
                total_correct += (output.argmax(dim=1) == target).sum().item()
                total_seen += target.size(0)

        avg_loss = total_loss / total_seen if total_seen > 0 else 0.0
        avg_acc = total_correct / total_seen if total_seen > 0 else 0.0
        return avg_loss, avg_acc

    def fit(self):
        for round_idx in range(1, self.num_rounds + 1):
            print(f"[WW-FL] Round {round_idx}/{self.num_rounds}")

            cluster_states = []
            cluster_sizes = []
            cluster_losses = []
            cluster_accs = []

            for cluster_id in sorted(self.cluster_map.keys()):
                selected_client_ids = self._sample_clients(cluster_id, round_idx)
                cluster_dataset = self._get_cluster_dataset(cluster_id, selected_client_ids)

                state_dict, cluster_size, train_loss, train_acc = self._train_cluster_model(cluster_dataset)

                cluster_states.append(state_dict)
                cluster_sizes.append(cluster_size)
                cluster_losses.append(train_loss)
                cluster_accs.append(train_acc)

            self._aggregate_cluster_models(cluster_states, cluster_sizes)
            test_loss, test_acc = self.evaluate()

            round_info = {
                "round": round_idx,
                "cluster_train_loss": sum(cluster_losses) / len(cluster_losses),
                "cluster_train_acc": sum(cluster_accs) / len(cluster_accs),
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
            self.history.append(round_info)

            print(
                f"cluster_loss={round_info['cluster_train_loss']:.4f} "
                f"cluster_acc={round_info['cluster_train_acc']:.4f} "
                f"test_loss={test_loss:.4f} "
                f"test_acc={test_acc:.4f}"
            )

        return self.global_model, self.history