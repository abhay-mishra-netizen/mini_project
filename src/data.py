import random
from typing import Dict, List

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def get_cifar10_datasets(data_dir="./data"):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    return train_dataset, test_dataset


def split_dataset_into_clients(
    dataset: Dataset,
    num_clients: int,
    samples_per_client: int,
    seed: int = 42,
):
    random.seed(seed)
    total_samples = len(dataset)
    client_datasets: Dict[int, Dataset] = {}

    for client_id in range(num_clients):
        indices = [random.randrange(total_samples) for _ in range(samples_per_client)]
        client_datasets[client_id] = Subset(dataset, indices)

    return client_datasets


def build_cluster_map(num_clients: int, num_clusters: int):
    if num_clients % num_clusters != 0:
        raise ValueError("num_clients must be divisible by num_clusters")

    clients_per_cluster = num_clients // num_clusters
    cluster_map: Dict[int, List[int]] = {}

    start = 0
    for cluster_id in range(num_clusters):
        end = start + clients_per_cluster
        cluster_map[cluster_id] = list(range(start, end))
        start = end

    return cluster_map


def make_client_loaders(
    client_datasets,
    batch_size,
    num_workers=2,
    shuffle=True,
    pin_memory=True,
):
    loaders = []

    for client_id in sorted(client_datasets.keys()):
        loader = DataLoader(
            client_datasets[client_id],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        loaders.append(loader)

    return loaders


def make_test_loader(
    test_dataset,
    batch_size=256,
    num_workers=2,
    pin_memory=True,
):
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_fl_setup(
    data_dir="./data",
    num_clients=1000,
    samples_per_client=200,
    batch_size=64,
    test_batch_size=256,
    num_workers=2,
    seed=42,
):
    train_dataset, test_dataset = get_cifar10_datasets(data_dir)

    client_datasets = split_dataset_into_clients(
        dataset=train_dataset,
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        seed=seed,
    )

    client_loaders = make_client_loaders(
        client_datasets=client_datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    test_loader = make_test_loader(
        test_dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return client_loaders, test_loader


def get_wwfl_setup(
    data_dir="./data",
    num_clients=1000,
    num_clusters=10,
    samples_per_client=200,
    test_batch_size=256,
    num_workers=2,
    seed=42,
):
    train_dataset, test_dataset = get_cifar10_datasets(data_dir)

    client_datasets = split_dataset_into_clients(
        dataset=train_dataset,
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        seed=seed,
    )

    cluster_map = build_cluster_map(
        num_clients=num_clients,
        num_clusters=num_clusters,
    )

    test_loader = make_test_loader(
        test_dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return client_datasets, cluster_map, test_loader