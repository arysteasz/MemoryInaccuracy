import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import cwt, ricker


csv_fname = "flow1.csv"

def ReadData (csv_path):
    c1 = []
    with open(csv_fname, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            c1.append(row[0])

    data = np.asarray(c1, dtype=np.float32)
    Fs = 200.0
    t0 = 0.0
    t1 = len(data) / Fs
    N = int((t1 - t0) * Fs)
    timedata = np.linspace(t0, t1, N)


    flowdata=np.array([timedata, data])

    return flowdata

def PlotData(a_data, b_data):
    #plt.scatter(a_data, b_data)
    plt.loglog(a_data, b_data)
    plt.xlabel('time data[s]')
    plt.ylabel('memory jump')
    plt.show()


def ContinousWaveltTrans(jump_data):
    Fs = 200.0
    t0 = 0.0
    t1 = len(jump_data) / Fs
    N = int((t1 - t0) * Fs)
    t = np.linspace(t0, t1, N)

    widths1 = np.arange(0.1, 10.0, 0.1)
    cwtmatr1 = cwt(jump_data, ricker, widths1)
    plt.pcolor(t, widths1, cwtmatr1)
    plt.show()

    return(cwtmatr1)

def Spectogram(jump_data):
    NFFT = 1024
    Fs = 200.0
    t0 = 0.0
    t1 = len(jump_data) / Fs
    N = int((t1 - t0) * Fs)
    t = np.linspace(t0, t1, N)
    Pxx, freqs, bins, im = plt.specgram(jump_data, NFFT=NFFT, Fs=Fs, noverlap=900)
    plt.show()

    return(Pxx)

ReadData(csv_fname)
time_data = ReadData(csv_fname)[0,:]
jump_data = ReadData(csv_fname)[1,:]

PlotData(time_data, jump_data)
ContinousWaveltTrans(jump_data)
Spectogram(jump_data)
