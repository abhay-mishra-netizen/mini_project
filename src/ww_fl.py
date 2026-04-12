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
        dataset_name='mnist'
    ):
        self.model_fn = model_fn
        self.client_datasets = client_datasets
        self.cluster_map = cluster_map
        self.test_loader = test_loader
        self.num_rounds = num_rounds
        self.clients_per_cluster_per_round = \
            clients_per_cluster_per_round
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed
        self.dataset_name = dataset_name

        # global model lives on global servers
        # in real WW-FL this would be secret shared
        # in simulation we keep it plaintext
        self.global_model = self.model_fn().to(device)

        # cluster memory accumulates data across rounds
        # matching paper Algorithm 1 line 9:
        # Dt = current data UNION previous data
        self.cluster_memory = {
            cluster_id: []
            for cluster_id in self.cluster_map
        }
        self.history = []

    def _sample_clients(self, cluster_id, round_idx):
        # reproducible random sampling
        # matching paper: random subset selected each round
        rng = random.Random(
            self.seed + 1000 * round_idx + cluster_id
        )
        client_ids = self.cluster_map[cluster_id]
        k = min(
            self.clients_per_cluster_per_round,
            len(client_ids)
        )
        return rng.sample(client_ids, k)

    def _get_cluster_dataset(self, cluster_id,
                              selected_client_ids):
        # clients share data to cluster servers
        # simulates the Share functionality from paper
        # data stays within cluster trust zone
        new_parts = [
            self.client_datasets[cid]
            for cid in selected_client_ids
        ]
        self.cluster_memory[cluster_id].extend(new_parts)

        # prevent RAM overflow over long training
        # keep last 20 rounds worth of data
        max_parts = self.clients_per_cluster_per_round * 20
        if len(self.cluster_memory[cluster_id]) > max_parts:
            self.cluster_memory[cluster_id] = \
                self.cluster_memory[cluster_id][-max_parts:]

        if len(self.cluster_memory[cluster_id]) == 1:
            return self.cluster_memory[cluster_id][0]
        return ConcatDataset(self.cluster_memory[cluster_id])

    def _train_cluster_model(self, cluster_dataset):
        # simulates the Train functionality from paper
        # in real WW-FL this runs under MPC encryption
        # in simulation mode this is plaintext training
        # which gives identical accuracy results
        # as confirmed by paper Figure 4 (less than 0.1%
        # difference between plaintext and CrypTen)

        model = self.model_fn().to(self.device)

        # load global weights into cluster model
        # simulates Reshare from global to cluster servers
        model.load_state_dict(
            copy.deepcopy(self.global_model.state_dict())
        )
        model.train()

        # pool all client data for this cluster
        # this is the key WW-FL difference from plain FL
        # clients contribute data to cluster pool
        # cluster trains on combined data
        # larger batch size (80 vs 8 in FL) reflects this
        loader = DataLoader(
            cluster_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device == 'cuda')
        )

        optimizer = optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=0.9
        )
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
                total_correct += (
                    output.argmax(dim=1) == target
                ).sum().item()
                total_seen += target.size(0)

        avg_loss = total_loss / total_seen \
                   if total_seen > 0 else 0.0
        avg_acc = total_correct / total_seen \
                  if total_seen > 0 else 0.0

        # return weights to global servers
        # simulates Reshare from cluster back to global
        return (
            copy.deepcopy(model.state_dict()),
            len(cluster_dataset),
            avg_loss,
            avg_acc
        )

    def _aggregate_cluster_models(self, cluster_states,
                                   cluster_sizes):
        # simulates the Agg functionality from paper
        # global servers aggregate cluster models
        # weighted by cluster dataset size
        # in real WW-FL this happens on secret shares
        # giving identical results to this plaintext version
        new_state = copy.deepcopy(cluster_states[0])
        total_size = sum(cluster_sizes)

        for key in new_state.keys():
            new_state[key] = cluster_states[0][key] * \
                             (cluster_sizes[0] / total_size)
            for i in range(1, len(cluster_states)):
                new_state[key] += \
                    cluster_states[i][key] * \
                    (cluster_sizes[i] / total_size)

        # update global model
        # in real WW-FL global model stays secret shared
        # never revealed to cluster servers or clients
        self.global_model.load_state_dict(new_state)

    def evaluate(self):
        # evaluation on test set
        # simulates secure inference from paper section 3.4
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
                total_correct += (
                    output.argmax(dim=1) == target
                ).sum().item()
                total_seen += target.size(0)

        avg_loss = total_loss / total_seen \
                   if total_seen > 0 else 0.0
        avg_acc = total_correct / total_seen \
                  if total_seen > 0 else 0.0
        return avg_loss, avg_acc

    def fit(self):
        for round_idx in range(1, self.num_rounds + 1):
            print(
                f"[WW-FL] Round {round_idx}/{self.num_rounds}"
            )

            cluster_states = []
            cluster_sizes = []
            cluster_losses = []
            cluster_accs = []

            # each cluster trains independently
            # this can be parallelized in real deployment
            for cluster_id in sorted(
                self.cluster_map.keys()
            ):
                selected_client_ids = self._sample_clients(
                    cluster_id, round_idx
                )
                cluster_dataset = self._get_cluster_dataset(
                    cluster_id, selected_client_ids
                )

                state_dict, cluster_size, \
                train_loss, train_acc = \
                    self._train_cluster_model(
                        cluster_dataset
                    )

                cluster_states.append(state_dict)
                cluster_sizes.append(cluster_size)
                cluster_losses.append(train_loss)
                cluster_accs.append(train_acc)

            # global aggregation after all clusters finish
            self._aggregate_cluster_models(
                cluster_states, cluster_sizes
            )
            test_loss, test_acc = self.evaluate()

            round_info = {
                "round": round_idx,
                "cluster_train_loss": sum(cluster_losses) /
                                      len(cluster_losses),
                "cluster_train_acc": sum(cluster_accs) /
                                     len(cluster_accs),
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
            self.history.append(round_info)

            print(
                f"cluster_loss="
                f"{round_info['cluster_train_loss']:.4f} "
                f"cluster_acc="
                f"{round_info['cluster_train_acc']:.4f} "
                f"test_loss={test_loss:.4f} "
                f"test_acc={test_acc:.4f}"
            )

        return self.global_model, self.history