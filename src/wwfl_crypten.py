import copy
import random
from typing import Dict, List, Tuple

import crypten
import torch
import torch.nn.functional as F
from crypten.config import cfg
from models import get_model
from torch.utils.data import DataLoader


cfg.mpc.provider = "TFP"
cfg.encoder.precision_bits = 22


def _build_model(model_name):
    return get_model(model_name)


def _normalize_state_dict(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        normalized_key = key[:-5] if key.endswith(".data") else key
        normalized[normalized_key] = value
    return normalized


class WWFLCrypTenTrainer:
    def __init__(
        self,
        model_name,
        client_datasets,
        cluster_map,
        test_loader,
        num_rounds=120,
        clients_per_cluster_per_round=4,
        local_epochs=2,
        batch_size=32,
        lr=0.02,
        memory_rounds=2,
        seed=42,
        progress_interval=1,
    ):
        self.model_name = model_name
        self.client_datasets = client_datasets
        self.cluster_map = cluster_map
        self.test_loader = test_loader
        self.num_rounds = num_rounds
        self.clients_per_cluster_per_round = clients_per_cluster_per_round
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.memory_rounds = memory_rounds
        self.seed = seed
        self.progress_interval = progress_interval

        if not crypten.is_initialized():
            crypten.init()

        self.global_model = _build_model(self.model_name).cpu()
        self.history = []
        self.client_tensors = self._materialize_client_datasets()
        self.cluster_memory: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {
            cluster_id: [] for cluster_id in cluster_map
        }

    def _materialize_client_datasets(self):
        materialized = {}
        for client_id, dataset in self.client_datasets.items():
            features = []
            labels = []
            loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
            for batch_x, batch_y in loader:
                features.append(batch_x.cpu())
                labels.append(batch_y.cpu())
            materialized[client_id] = (
                torch.cat(features, dim=0),
                torch.cat(labels, dim=0),
            )
        return materialized

    def _sample_clients(self, cluster_id, round_idx):
        rng = random.Random(self.seed + 1000 * round_idx + cluster_id)
        candidates = self.cluster_map[cluster_id]
        sample_size = min(self.clients_per_cluster_per_round, len(candidates))
        return rng.sample(candidates, sample_size)

    def _get_cluster_training_data(self, cluster_id, selected_client_ids):
        new_chunks = [self.client_tensors[client_id] for client_id in selected_client_ids]
        self.cluster_memory[cluster_id].extend(new_chunks)

        max_chunks = self.clients_per_cluster_per_round * self.memory_rounds
        if len(self.cluster_memory[cluster_id]) > max_chunks:
            self.cluster_memory[cluster_id] = self.cluster_memory[cluster_id][-max_chunks:]

        features = torch.cat([chunk[0] for chunk in self.cluster_memory[cluster_id]], dim=0)
        labels = torch.cat([chunk[1] for chunk in self.cluster_memory[cluster_id]], dim=0)
        permutation = torch.randperm(features.size(0))
        return features[permutation], labels[permutation]

    def _encrypt_model(self, state_dict, sample_shape, fixed_batch_size):
        plaintext_model = _build_model(self.model_name).cpu().eval()
        plaintext_model.load_state_dict(copy.deepcopy(state_dict))
        dummy_input = torch.zeros(fixed_batch_size, *sample_shape, dtype=torch.float32)
        private_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)
        private_model.encrypt(src=0)
        private_model.train()
        return private_model

    def _train_cluster_model(self, cluster_id, selected_client_ids):
        train_x, train_y = self._get_cluster_training_data(cluster_id, selected_client_ids)
        fixed_batch_size = min(self.batch_size, train_x.size(0))
        private_model = self._encrypt_model(
            self.global_model.state_dict(),
            train_x.shape[1:],
            fixed_batch_size,
        )
        criterion = crypten.nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0.0
        total_seen = 0

        for _ in range(self.local_epochs):
            for start in range(0, train_x.size(0), fixed_batch_size):
                end = min(start + fixed_batch_size, train_x.size(0))
                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                true_batch_size = batch_y.size(0)

                if true_batch_size < fixed_batch_size:
                    pad_count = fixed_batch_size - true_batch_size
                    batch_x = torch.cat([batch_x, batch_x[:pad_count]], dim=0)
                    batch_y = torch.cat([batch_y, batch_y[:pad_count]], dim=0)

                encrypted_x = crypten.cryptensor(batch_x, src=0)
                encrypted_y = crypten.cryptensor(
                    F.one_hot(batch_y, num_classes=10).float(),
                    src=0,
                )

                private_model.zero_grad()
                predictions = private_model(encrypted_x)
                loss = criterion(predictions, encrypted_y)
                loss.backward()
                private_model.update_parameters(self.lr)

                total_loss += float(loss.get_plain_text().item()) * true_batch_size
                predicted_labels = predictions.get_plain_text().argmax(dim=1)[:true_batch_size]
                total_correct += float((predicted_labels == batch_y[:true_batch_size]).sum().item())
                total_seen += true_batch_size

        private_model.decrypt()
        return (
            _normalize_state_dict(copy.deepcopy(private_model.state_dict())),
            train_x.size(0),
            total_loss / total_seen if total_seen else 0.0,
            total_correct / total_seen if total_seen else 0.0,
        )

    def _aggregate_cluster_models(self, cluster_states, cluster_sizes):
        total_weight = float(sum(cluster_sizes))
        aggregated_state = copy.deepcopy(cluster_states[0])

        for key in aggregated_state.keys():
            encrypted_sum = None
            for idx, state in enumerate(cluster_states):
                weighted_tensor = state[key] * (cluster_sizes[idx] / total_weight)
                encrypted_tensor = crypten.cryptensor(weighted_tensor, src=0)
                encrypted_sum = encrypted_tensor if encrypted_sum is None else encrypted_sum + encrypted_tensor
            aggregated_state[key] = encrypted_sum.get_plain_text()

        self.global_model.load_state_dict(aggregated_state)

    def evaluate(self):
        self.global_model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                logits = self.global_model(data.cpu())
                loss = criterion(logits, target.cpu())
                total_loss += loss.item() * target.size(0)
                total_correct += (logits.argmax(dim=1) == target.cpu()).sum().item()
                total_seen += target.size(0)

        avg_loss = total_loss / total_seen if total_seen else 0.0
        avg_acc = total_correct / total_seen if total_seen else 0.0
        return avg_loss, avg_acc

    def fit(self):
        for round_idx in range(1, self.num_rounds + 1):
            print(f"[WWFL-CrypTen] Round {round_idx}/{self.num_rounds}")

            cluster_states = []
            cluster_sizes = []
            cluster_losses = []
            cluster_accs = []

            for cluster_id in sorted(self.cluster_map.keys()):
                selected_clients = self._sample_clients(cluster_id, round_idx)
                state_dict, cluster_size, train_loss, train_acc = self._train_cluster_model(
                    cluster_id, selected_clients
                )
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

            if round_idx % self.progress_interval == 0 or round_idx == self.num_rounds:
                print(
                    "cluster_loss={:.4f} cluster_acc={:.4f} test_loss={:.4f} test_acc={:.4f}".format(
                        round_info["cluster_train_loss"],
                        round_info["cluster_train_acc"],
                        test_loss,
                        test_acc,
                    )
                )

        return self.global_model, self.history


def get_recommended_wwfl_crypten_config():
    return {
        "dataset": "mnist",
        "model": "lenet",
        "num_clients": 120,
        "num_clusters": 6,
        "clients_per_cluster_per_round": 4,
        "samples_per_client": 200,
        "rounds": 120,
        "local_epochs": 2,
        "batch_size": 32,
        "learning_rate": 0.02,
        "memory_rounds": 2,
        "test_batch_size": 256,
    }
