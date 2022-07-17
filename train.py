import time
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from AdaptiveWeightNetwork.utils import Logger, LoadDataset
from AdaptiveWeightNetwork.network import AdaptiveWeightNetwork, HrEstimationNetwork

# start_time is the time when the process begin
start_time = time.time()

# save the standard output in train_valid.log file
sys.stdout = Logger("./logs/train.log")


def train_model(model, loss_function, optimizer, epochs, model_path, model_name):
    loss_data = []
    best_loss = sys.maxsize
    best_epoch = 0

    for epoch in range(epochs):
        print("==================epoch : {}/{} ==================".format(epoch + 1, epochs))

        epoch_start = time.time()
        model.train()

        train_loss = 0.0
        valid_loss = 0.0

        print('======train=====')
        for batch_index, train_data in enumerate(train_loader, 0):
            inputs, labels = train_data
            inputs = inputs.to(device)
            print('     inputs:\n' + str(inputs.int()))
            labels = labels.to(device)
            print('     labels:\n' + str(labels.int()))
            optimizer.zero_grad()
            labels = labels.to(torch.float32)
            outputs = model(inputs)
            print('     outputs:\n' + str(torch.squeeze(outputs)))
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()
            print('======valid=====')
            for batch_size, valid_data in enumerate(valid_loader, 0):
                inputs, labels = valid_data
                inputs = inputs.to(device)
                print('     inputs:\n' + str(inputs.int()))
                labels = labels.to(device)
                print('     labels:\n' + str(labels.int()))
                outputs = model(inputs)
                print('     outputs:\n' + str(torch.squeeze(outputs)))
                outputs = torch.squeeze(outputs)
                loss = loss_function(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

        # compute loss
        avg_train_loss = train_loss / train_data_size
        avg_valid_loss = valid_loss / valid_data_size

        loss_data.append([avg_train_loss, avg_valid_loss])

        if best_loss > avg_valid_loss:
            best_loss = avg_valid_loss
            best_epoch = epoch + 1
            # save the model which has lowest loss
            torch.save(model.state_dict(), model_path + 'best-' + model_name)

        epoch_end = time.time()

        print("==========epoch end==========\n"
              "Epoch: {:03d}, Training: Loss: {:.4f} \n Validation: Loss: {:.4f}, Time: {:.4f}s".format(
            epoch + 1, avg_train_loss, avg_valid_loss, epoch_end - epoch_start
        ))
        print("lowest loss for validation : {:.4f} at epoch {:03d}".format(best_loss, best_epoch))
        torch.save(model.state_dict(), model_path + str(epoch + 1) + '-' + model_name)

    return model, loss_data

# plot_loss plots the loss curve
def plot_loss(loss_data, loss_curve):
    loss_data = np.array(loss_data)
    plt.plot(range(20, num_epochs), loss_data[:, 0:2][20:num_epochs])
    plt.legend(['train loss', 'valid loss'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('epoch: 20~num_epochs loss curve')
    plt.savefig(loss_curve)
    plt.show()

# If you want use HR estimation network, replace AdaptiveWeightNetwork() with HrEstimationNetwork()
model = AdaptiveWeightNetwork()
# print model structure
print(model)
# GPU is used for training by default
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss(reduction='mean')
batch_size = 64

train_transforms = transforms.Compose([transforms.ToTensor()])
test_valid_transforms = transforms.Compose([transforms.ToTensor()])
train_dataset = LoadDataset('','', transform=train_transforms)
valid_dataset = LoadDataset('','', transform=test_valid_transforms)
# train_loader loads the training dataset,　you can choose to shuffle your training dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# valid_loader loads the validation dataset
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

train_data_size = len(train_dataset)
valid_data_size = len(valid_dataset)

num_epochs = 150
model_save_path = ""
last_epoch_model_save_path = ""
trained_model, acc_loss_data = train_model(model, criterion, optimizer, num_epochs, model_save_path)
torch.save(trained_model.state_dict(), last_epoch_model_save_path)

loss_curve_save_path = ""
plot_loss(acc_loss_data, loss_curve_save_path)

end_time = time.time()
# print the running time
print("training and validation costs: {:.4f}s".format(end_time - start_time))
