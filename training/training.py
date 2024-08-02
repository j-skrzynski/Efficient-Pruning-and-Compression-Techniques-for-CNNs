import torch
from sklearn.metrics import confusion_matrix

def one_epoch_train(net, optimizer, criterion, trainloader, epoch_id, device=None, l2_reg=0.0001):
    print("Selecting the device")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Computing on: " + str(device))
    
    net.train(True)
    running_loss = 0.0
    loss_total = 0.0

    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    net.to(device)  # Przenieś model na GPU

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Przenieś dane na GPU

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        
        # Compute the loss with L2 regularization
        loss = criterion(outputs, labels) + net.l2_loss()
        
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # print statistics
        running_loss += loss.item()
        loss_total += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch_id + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
    cm = confusion_matrix(all_labels, all_predictions)

    return loss_total / len(trainloader), correct / total, cm



def evaluate(net, criterion, testloader, device=None):
    print("Selecting the device")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Computing on: " + str(device))

    net.eval()
    correct = 0
    total = 0
    val_total = 0
    all_labels = []
    all_predictions = []

    net.to(device)  # Przenieś model na GPU

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # Przenieś dane na GPU

            # calculate outputs by running images through the network
            outputs = net(images)
            val_total += criterion(outputs, labels).item()
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions)

    return val_total / len(testloader), correct / total, cm