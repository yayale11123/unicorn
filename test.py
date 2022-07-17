import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from AdaptiveWeightNetwork.network import AdaptiveWeightNetwork
from AdaptiveWeightNetwork.utils import Logger, Bland_Altman_Plot, Scatter_Plot_Fit, LoadDataset

# save the standard output in train_valid.log file
sys.stdout = Logger("./logs/test.log")
# start_time is the time when the process begin


# If you want use HR estimation network, replace AdaptiveWeightNetwork() with HrEstimationNetwork()
model = AdaptiveWeightNetwork()
# load the trained model
trained_model_path = ''
model.load_state_dict(torch.load(trained_model_path), False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

test_transforms = transforms.Compose([transforms.ToTensor()])
test_dataset = LoadDataset('', '', transform=test_transforms)
test_data_size = len(test_dataset)
batch_size=64
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# features record weights
features = []
# register hook to obtain weights
def hook(module, input, output):
    features.append(output.clone().detach())

# plot four raw signals and the weighted signal
def get_weighted_signal_ECG(raw_signals,batch,i):
    plt.figure()
    fps = 30
    xticks = np.round(
        np.arange(0, len(raw_signals[i][2]) / fps + 0.001, len(raw_signals[i][2]) / fps / 10), 2)
    plt.xticks(xticks)
    plt.xlim([0, len(raw_signals[i][2]) / fps])
    x_xlim = np.round(np.arange(0, len(raw_signals[i][2]), 1) / fps, 2)

    plt.subplot(2, 3, 1)
    plt.plot(x_xlim, raw_signals[i][0], 'b')


    plt.subplot(2, 3, 2)
    plt.plot(x_xlim, raw_signals[i][1], 'g')

    weighted_signals = torch.squeeze(raw_signals, 1)
    x = features[batch]
    x = torch.squeeze(x, 2)
    x_weight = F.softmax(x, dim=1)
    x_weight = torch.unsqueeze(x_weight, 1)

    x_weight = x_weight.expand(raw_signals.shape)
    weighted_signals = weighted_signals.to(device)
    # 矩阵相乘加权
    weighted_signal = weighted_signals * x_weight
    weighted_signal = weighted_signal.permute(0, 2, 1)
    # 将加权后的信号相加
    weighted_signal = weighted_signal[:, 0, :] + weighted_signal[:, 1, :] + weighted_signal[:, 2, :] + weighted_signal[:, 3, :]
    print(('weights of batch {}, sample {} ' + str(x_weight[i].int())).format(batch, i))

    plt.subplot(2, 3, 3)
    plt.plot(x_xlim, weighted_signal.cpu()[i], 'y')

    plt.subplot(2, 3, 4)
    plt.plot(x_xlim, raw_signals[i][2], 'b')

    plt.subplot(2, 3, 5)
    plt.plot(x_xlim, raw_signals[i][3], 'g')

    plot_save_path = ''
    plt.savefig(plot_save_path)
    # plt.show()


def test(model):
    start_time = time.time()
    with torch.no_grad():  # 不计算梯度
        model.eval()

        predictions_array = np.array([])
        labels_array = np.array([])
        for batch, (inputs, labels) in enumerate(test_loader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # show output shape and hierarchical view of net
            # print(summary(model, inputs, show_input=False, show_hierarchical=True))

            handle = model.fc.register_forward_hook(hook)
            outputs = model(inputs)
            for i in range(labels.shape[0]):
                get_weighted_signal_ECG(inputs,batch,i)
            handle.remove()

            outputs = torch.squeeze(outputs)

            labels = labels.cpu().numpy()
            prediction = outputs.cpu().numpy()
            labels_array = np.hstack((labels_array, labels))
            predictions_array = np.hstack((predictions_array, prediction))

            print('labels:      ' + str(labels.astype(int)) + '\n' + 'prediction:  ' + str(
                prediction.astype(int)) + '\n' + 'error:       ' + str((
                abs(prediction.astype(int) - labels.astype(int)))))
            print('---------------------------------------')

        data = []
        data.append(labels_array)
        data.append(predictions_array)

        test_data_size = len(test_dataset)

        MEAN = np.mean(predictions_array - labels_array)
        print('MEAN: ' + str(round(MEAN, 2)))

        HRe = np.mean(predictions_array - labels_array)
        Std = np.sqrt(np.mean(np.power((predictions_array - labels_array) - HRe, 2)))
        print('STD: ' + str(round(Std, 2)))

        MAE = np.mean(np.abs(predictions_array - labels_array))
        print('MAE:' + str(round(MAE, 2)))

        RMSE = np.sqrt(np.sum(np.power(predictions_array - labels_array, 2)) / test_data_size)
        print('RMSE:' + str(round(RMSE, 2)))

        MAPE = np.mean((np.abs(predictions_array - labels_array) / labels_array))
        print('MAPE:' + str(round(MAPE * 100, 2)) + '%')

        a = np.mean(np.multiply(labels_array - np.mean(labels_array), predictions_array - np.mean(predictions_array)))
        b = np.sqrt(np.mean(np.power(labels_array - np.mean(labels_array), 2))) * np.sqrt(
            np.mean(np.power(predictions_array - np.mean(predictions_array), 2)))
        p = a / b
        print('pearson:' + round(p, 2))

    # plot and save bland-altman figure
    Bland_Altman_Plot(labels_array, predictions_array, path='')
    # plot and save scatter figure
    Scatter_Plot_Fit(labels_array, predictions_array, path='')

    end_time = time.time()
    # print the running time
    print("training and validation costs: {:.4f}s".format(end_time - start_time))
