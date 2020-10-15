import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def FW(kinematic_file, IMU_file):

    # read kinematic and IMU data
    kdata = pd.read_csv(kinematic_file)
    idata = pd.read_csv(IMU_file)

    # Kinematic data pre-processing:
    # low-pass filtering with 10 Hz threshold and Savitzky-Golay filter to smoothen graph
    fk = kdata['Time'].idxmax() / kdata['Time'].max()
    sos = signal.butter(1, 10, 'lp', fs = fk, output = 'sos')

    kdata['RHeel'] = signal.sosfiltfilt(sos,kdata['RHeel']) # apply low-pass filter
    kdata['RHeel'] = signal.savgol_filter(kdata['RHeel'], window_length = 51, polyorder = 3) # data smoothening
    kdata['LHeel'] = signal.sosfiltfilt(sos,kdata['LHeel'])
    kdata['LHeel'] = signal.savgol_filter(kdata['LHeel'], window_length = 51, polyorder = 3)

    # IMU data pre-processing:
    # heel accelerometer data: remove mean and apply low-pass filtering with 10 Hz threshold
    fi = idata['Time'].idxmax() / idata['Time'].max()
    sos = signal.butter(1, 10, 'lp', fs = fi, output = 'sos')
    for j in [i for i in idata.columns if 'H.ACC' in i]:
        idata[j] = idata[j] - idata[j].mean() # remove mean
        idata[j] = signal.sosfilt(sos, idata[j]) # apply low-pass filter

    # heel gyroscope data: apply band-pass filter with 0.001 - 5 Hz thresholds.
    sos = signal.butter(1, (0.001,5), 'bp', fs = fi, output = 'sos')
    for j in [i for i in idata.columns if 'H.GY' in i]:
        idata[j] = signal.sosfiltfilt(sos, idata[j])


    # RHS: Right heel strike; RTO: Right toes off; LHS: Left heel strike; LTO: Left toes off

    # identify maxima (foot contacts) and minima (foot clearance) points from kinematic data
    n=50 # distance between data points. this number can me modified, depending on recording frequency
    kdata['RHS'] = kdata.iloc[signal.argrelextrema(kdata['RHeel'].values, np.greater_equal, order = n)[0]]['RHeel']
    kdata['RTO'] = kdata.iloc[signal.argrelextrema(kdata['RHeel'].values, np.less_equal, order = n)[0]]['RHeel']
    kdata['LHS'] = kdata.iloc[signal.argrelextrema(kdata['LHeel'].values, np.greater_equal, order = n)[0]]['LHeel']
    kdata['LTO'] = kdata.iloc[signal.argrelextrema(kdata['LHeel'].values, np.less_equal, order = n)[0]]['LHeel']

    # identify gait events from IMU data: maxima points of heel accelerometer as heel strikes, and minima points of
    # heel gyroscope as toes of.
    n=250
    idata['RHS'] = idata.iloc[signal.argrelextrema(signal.savgol_filter(idata['RH.ACC.Z'], 35, 2),
                                            np.greater_equal,
                                            order = n)[0]]['RH.ACC.Z']
    idata['RTO'] = idata.iloc[signal.argrelextrema(signal.savgol_filter(idata['RH.GY.Z'].values, 51, 3),
                                            np.less_equal,
                                            order = n)[0]]['RH.GY.Z']
    idata['LHS'] = idata.iloc[signal.argrelextrema(signal.savgol_filter(idata['LH.ACC.Z'].values, 35, 2),
                                            np.greater_equal,
                                            order = n)[0]]['LH.ACC.Z']
    idata['LTO'] = idata.iloc[signal.argrelextrema(signal.savgol_filter(idata['LH.GY.Z'].values,51,3), # notice reversing of gyroscope data
                                            np.greater_equal,
                                            order = n)[0]]['LH.GY.Z']

    # display kinematic, accelerometer and gyroscope data, with identified gait events
    plt.figure (figsize = (15, 5))
    plt.subplot(2,1,1)
    plt.plot(kdata['Time'], kdata['RHeel'], alpha = 0.5)
    plt.plot(idata['Time'], signal.savgol_filter(idata['RH.ACC.Z'], 35, 2), alpha = 0.5)
    plt.plot(idata['Time'], signal.savgol_filter(idata['RH.GY.Z'] / 600, 51, 3), alpha = 0.5) # gyroscope data i recuded to fit in graph
    plt.scatter(idata['Time'], idata['RHS'],color = 'red')
    plt.scatter(idata['Time'], idata['RTO'] / 600,color = 'green')
    plt.scatter(kdata['Time'], kdata['RHS'], color = 'red',marker = 's')
    plt.scatter(kdata['Time'], kdata['RTO'], color = 'green',marker = 's')
    plt.legend(['Kinematic','Accelerometer','Gyroscope'],bbox_to_anchor = (1,1))

    plt.subplot(2,1,2)
    plt.plot(kdata['Time'], kdata['LHeel'], alpha = 0.5)
    plt.plot(idata['Time'], signal.savgol_filter(idata['LH.ACC.Z'], 35, 2), alpha = 0.5)
    plt.plot(idata['Time'], signal.savgol_filter(idata['LH.GY.Z'] / 600, 51, 3), alpha = 0.5) # gyroscope data i recuded to fit in graph
    plt.scatter(idata['Time'], idata['LHS'],color = 'red')
    plt.scatter(idata['Time'], idata['LTO'] / 600,color = 'green')
    plt.scatter(kdata['Time'], kdata['LHS'], color = 'red',marker = 's')
    plt.scatter(kdata['Time'], kdata['LTO'], color = 'green',marker = 's')

    # return eight arrays:  right kinematic-based foot contacts times, right kinematic-based foot clearance times,
    #                       left kinematic-based foot contacts times, left kinematic-based foot clearance times,
    #                       right IMU-based foot contacts times, right IMU-based foot clearance times,
    #                       left IMU-based foot contacts times and left IMU-based foot clearance times

    return (kdata[['Time','RHS']].dropna()['Time'].values, kdata[['Time','RTO']].dropna()['Time'].values,
            kdata[['Time','LHS']].dropna()['Time'].values, kdata[['Time','LTO']].dropna()['Time'].values,
            idata[['Time','RHS']].dropna()['Time'].values, idata[['Time','RTO']].dropna()['Time'].values,
            idata[['Time','LHS']].dropna()['Time'].values, idata[['Time','LTO']].dropna()['Time'].values)



FW(kinematic_file = 'FW Kinematic data (example).csv', IMU_file = 'FW IMU data (example).csv')
