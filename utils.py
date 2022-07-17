import sys
import numpy as np
from scipy import signal
from torch.utils.data import Dataset
from matplotlib import pyplot as plt


# Logger can save the standard output in log file
class Logger(object):
    def __init__(self, filePath="./logs/train_valid.log"):
        self.terminal = sys.stdout
        self.log = open(filePath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def Bland_Altman_Plot(data1, data2, path="", *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference
    pearson = np.corrcoef(data1, data2)
    # print("Pearson correlation coefficient:\n{}".format(pearson))
    # print("Mean of the difference:{}\nStandard deviation of the difference:{}".format(md, sd))

    plt.cla()
    plt.scatter(mean, diff, c='red', s=10, *args, **kwargs)
    h1 = plt.axhline(md, color='black', linestyle='-')
    h2 = plt.axhline(md + 1.96 * sd, color='black', linestyle='--')
    h3 = plt.axhline(md - 1.96 * sd, color='black', linestyle='--')

    # plt.title('Bland-Altman Plot', fontsize=25)
    plt.xlim((min(mean) - 5, max(mean) + 5))
    plt.ylim((min(diff) - 5, max(diff) + 5))
    plt.xlabel('($\mathregular{HR_{label}}$ + $\mathregular{HR_{predict}}$)/2(bpm)', fontsize=15)
    plt.ylabel('$\mathregular{HR_{label}}$ - $\mathregular{HR_{predict}}(bpm)$', fontsize=15)
    plt.legend((h1, h2, h3), ('mean', 'mean+1.96δ', 'mean-1.96δ'))
    if path != "":
        plt.savefig(path)
    plt.show()


def Linear_Fit(data1, data2):
    z1 = np.polyfit(data1, data2, 1)  # 一次多项式拟合，相当于线性拟合
    return z1


def Scatter_Plot_Fit(data1, data2, path=""):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    plt.cla()
    plt.scatter(data1, data2, c='red', alpha=0.5, s=5)

    # plt.title('Scatter Plot', fontsize=25)
    plt.xlim((min(data1) - 5, max(data1) + 5))
    plt.ylim((min(data2) - 5, max(data2) + 5))
    plt.xlabel('$\mathregular{HR_{label}(bpm)}$', fontsize=15)
    plt.ylabel('$\mathregular{HR_{predict}(bpm)}$', fontsize=15)

    k, b = Linear_Fit(data1, data2)
    x = np.arange(min(data1) - 2.5, max(data1) + 2.5, 1)
    y = k * x + b
    plt.plot(x, y, 'k-')
    if path != "":
        plt.savefig(path)
    plt.show()


def show_signal(signal, fps, color, save_path='', title='', show=False, off=False):
    plt.figure(figsize=(16, 10))
    plt.xlabel("Time(s)", fontsize=35)

    xticks = np.round(np.arange(0, len(signal) / fps + 0.001, len(signal) / fps / 10), 2)
    plt.xticks(xticks, fontsize=40)
    plt.xlim([0, len(signal) / fps])
    # plt.xticks([])
    plt.yticks([])
    # plt.yticks([],linewidth=3)

    x = np.round(np.arange(0, len(signal), 1) / fps, 2)
    plt.plot(x, signal, color, linewidth=3)

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth('5')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if off == True:
        plt.axis('off')
    if title != '':
        plt.title(title)
    if save_path != '':
        plt.savefig(save_path)
    if show == True:
        plt.show()
    plt.cla()
    plt.close("all")

# bandpass_filter filters the signal, the cut-off frequency is [0.7-3] Hz.
def bandpass_filter(raw_signal,fps):
    singal_mean = np.mean(raw_signal)
    b, a = signal.butter(8, 0.7 * 2 / fps, 'highpass')
    raw_signal = signal.filtfilt(b, a, raw_signal)
    raw_signal = raw_signal + singal_mean
    b1, a1 = signal.butter(8, 3 * 2 / fps, 'lowpass')
    raw_signal = signal.filtfilt(b1, a1, raw_signal)
    return raw_signal

# reWrite LoadDataset to load your own dataset
class LoadDataset(Dataset):
    def __init__(self, dataset_file, transform=None):
        fh = open(dataset_file, 'r')
        signals = []
        for line in fh:
            line = line.rstrip()
            split = line.split()
            signals.append((split[0], float(split[1])))
            self.signals = signals
            self.transform = transform

    def __getitem__(self, index):
        fn, label = self.signals[index]
        signal = np.load(fn, allow_pickle=True)[0]
        if self.transform is not None:
            signal = self.transform(signal)
        return signal, label

    def __len__(self):
        return len(self.signals)
