import copy

import torch
import torch.nn as nn
import torch.optim as optim


def local_train(model, dataloader, epochs, lr=0.01, device="cuda"):
    model = model.to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for _ in range(epochs):
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)

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

    return copy.deepcopy(model.state_dict()), avg_loss, avg_acc, total_seen


def average_weights(weights_list, data_sizes):
    avg_weights = copy.deepcopy(weights_list[0])
    total_data = sum(data_sizes)

    for key in avg_weights.keys():
        avg_weights[key] = weights_list[0][key] * (data_sizes[0] / total_data)

        for i in range(1, len(weights_list)):
            avg_weights[key] += weights_list[i][key] * (data_sizes[i] / total_data)

    return avg_weights


def fedavg(global_model, client_loaders, num_rounds, local_epochs, lr, device):
    global_model = global_model.to(device)
    history = []

    for round_idx in range(num_rounds):
        print(f"[FL] Round {round_idx + 1}/{num_rounds}")

        local_weights = []
        data_sizes = []
        round_losses = []
        round_accs = []

        for loader in client_loaders:
            local_model = copy.deepcopy(global_model)

            weights, train_loss, train_acc, num_samples = local_train(
                local_model,
                loader,
                epochs=local_epochs,
                lr=lr,
                device=device,
            )

            local_weights.append(weights)
            data_sizes.append(num_samples)
            round_losses.append(train_loss)
            round_accs.append(train_acc)

        new_weights = average_weights(local_weights, data_sizes)
        global_model.load_state_dict(new_weights)

        history.append({
            "round": round_idx + 1,
            "train_loss": sum(round_losses) / len(round_losses),
            "train_acc": sum(round_accs) / len(round_accs),
        })

    return global_model, history


def evaluate(model, dataloader, device="cpu"):
    model.eval()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)
            loss = criterion(outputs, target)

            total_loss += loss.item() * target.size(0)
            correct += (outputs.argmax(dim=1) == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy