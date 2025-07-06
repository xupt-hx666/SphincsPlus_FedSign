import torch
from torch import nn, optim
from data_process import load_data


def train(args, model, client_id):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.base_layers.parameters(), 'lr': args.lr},
        {'params': model.personal_layers.parameters(), 'lr': args.lr * 1.2}
    ], weight_decay=args.weight_decay)

    train_loader, _, _ = load_data(client_id)

    for epoch in range(args.E):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(args.device)
            if labels.dim() > 1:
                labels = labels.squeeze(1)
            labels = labels.long().to(args.device)
            """nn.CrossEntropyLoss 要求标签必须是整数类型,long()就是将浮点类型转换成整数型张量"""
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_threshold)
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(
            f"Client {client_id} Epoch {epoch + 1}/{args.E} | Loss: {epoch_loss / len(train_loader):.4f} | Acc: {accuracy:.2f}%")

    return model


def test(args, model, client_id):
    model.eval()
    _, _, test_loader = load_data(client_id)
    criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(args.device)
            if labels.dim() > 1:
                labels = labels.squeeze(1)
            labels = labels.long().to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Client {client_id} Test | Loss: {test_loss / len(test_loader):.4f} | Acc: {accuracy:.2f}%")


def validate(args, model, client_id):
    model.eval()
    _, val_loader, _ = load_data(client_id)

    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(args.device)
            if labels.dim() > 1:
                labels = labels.squeeze(1)
            labels = labels.long().to(args.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


