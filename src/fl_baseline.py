import torch
import torch.nn as nn
import torch.optim as optim
import copy

def local_train( model , dataloader , epochs , lr = 0.01 , device = "cuda") :
    model.train()
    model.to(device)
    optimizer = optim.SGD( model.parameter() , lr=lr)
    criterion = nn.CrossEntropyLoss()
    for i in range(epochs) :
        for data , target in dataloader :
            data , target = data.to(device) , target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion( output , target )
        loss.backward()
        optimizer.step()
        return model.state_dict()

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
           for r in range(num_rounds) :
                print(f"Round {r+1}/{num_rounds}")
                local_weights = []
           for loader in client_loaders:
            local_model = copy.deepcopy(global_model)

            weights = local_train(
                local_model,
                loader,
                epochs=local_epochs,
                lr=lr,
                device=device
            )
            local_weights.append(weights)
            new_weights = average_weights(local_weights)
            global_model.load_state_dict(new_weights)
            return global_model
           
def evaluate(model, dataloader, device="cpu"):
    model.eval()
    model = model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy          