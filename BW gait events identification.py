import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def BW(kinematic_file, IMU_file):

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

#     IMU data pre-processing:
#     toes accelerometer data: remove mean and apply low-pass filtering with 10 Hz threshold
    fi = idata['Time'].idxmax() / idata['Time'].max()
    sos = signal.butter(1, 10, 'lp', fs = fi, output = 'sos')
    for j in [i for i in idata.columns if 'T.ACC' in i]:
        idata[j] = idata[j] - idata[j].mean() # remove mean
        idata[j] = signal.sosfilt(sos, idata[j]) # apply low-pass filter


#     RTS: Right foot contact; RHO: Right foot clearance; LTS: Left foot contact; LHO: Left foot clearance
#     identify minima (foot contacts) and maxima (foot clearance) points from kinematic data
    n=50 # distance between data points. this number can me modified, depending on recording frequency
    kdata['RTS'] = kdata.iloc[signal.argrelextrema(kdata['RTOE'].values, np.less_equal, order = n)[0]]['RTOE']
    kdata['RHO'] = kdata.iloc[signal.argrelextrema(kdata['RTOE'].values, np.greater_equal, order = n)[0]]['RTOE']
    kdata['LTS'] = kdata.iloc[signal.argrelextrema(kdata['LTOE'].values, np.less_equal, order = n)[0]]['LTOE']
    kdata['LHO'] = kdata.iloc[signal.argrelextrema(kdata['LTOE'].values, np.greater_equal, order = n)[0]]['LTOE']


#     identify foot contact as the maxima points of the toes acceleromter
    n = 250
    idata['RTS'] = idata.iloc[signal.argrelextrema(idata['RT.ACC.Z'].values, np.greater_equal, order = n)[0]]['RT.ACC.Z']
    idata['LTS'] = idata.iloc[signal.argrelextrema(idata['LT.ACC.Z'].values, np.greater_equal, order = n)[0]]['LT.ACC.Z']

#     identify foot clearance as a binary function - consecutive data points which are less than minus standard deviation
    t = signal.savgol_filter(idata['RT.ACC.Z'], window_length = 51, polyorder = 1)
    t = (t < -t.std())
    peaks = signal.peak_widths(t, signal.find_peaks(t, distance = n)[0])[2].astype('int')
    idata['RHO'] = np.array([np.nan for i in range (len(t))])
    idata['RHO'].loc[peaks] = idata['RT.ACC.Z'].loc[peaks]

    t = signal.savgol_filter(idata['LT.ACC.Z'], window_length = 51, polyorder = 1)
    t = (t < -t.std())
    peaks = signal.peak_widths(t, signal.find_peaks(t, distance = n)[0])[2].astype('int')
    idata['LHO'] = np.array([np.nan for i in range (len(t))])
    idata['LHO'].loc[peaks] = idata['LT.ACC.Z'].loc[peaks]



#     display kinematic and accelerometer, with identified gait events
    plt.figure (figsize = (15, 5))
    plt.subplot(2,1,1)
    plt.plot(kdata['Time'], kdata['RTOE'] * 5, alpha = 0.5) # kinematic data was multiplied by 5 for better visualization
    plt.plot(idata['Time'], idata['RT.ACC.Z'], alpha = 0.5)
    plt.scatter(idata['Time'], idata['RTS'], color = 'red')
    plt.scatter(idata['Time'], idata['RHO'], color = 'green')
    plt.scatter(kdata['Time'], kdata['RTS'] * 5, color = 'red',marker = 's')
    plt.scatter(kdata['Time'], kdata['RHO'] * 5, color = 'green',marker = 's')
    plt.legend(['Kinematic','Accelerometer'],bbox_to_anchor = (1,1))

    plt.subplot(2,1,2)
    plt.plot(kdata['Time'], kdata['LTOE'] * 5, alpha = 0.5)
    plt.plot(idata['Time'], idata['LT.ACC.Z'], alpha = 0.5)
    plt.scatter(idata['Time'], idata['LTS'],color = 'red')
    plt.scatter(idata['Time'], idata['LHO'],color = 'green')
    plt.scatter(kdata['Time'], kdata['LTS'] * 5, color = 'red',marker = 's')
    plt.scatter(kdata['Time'], kdata['LHO'] * 5, color = 'green',marker = 's')

    # return eight arrays:  right kinematic-based foot contacts times, right kinematic-based foot clearance times,
    #                       left kinematic-based foot contacts times, left kinematic-based foot clearance times,
    #                       right IMU-based foot contacts times, right IMU-based foot clearance times,
    #                       left IMU-based foot contacts times and left IMU-based foot clearance times

    return (kdata[['Time','RTS']].dropna()['Time'].values, kdata[['Time','RHO']].dropna()['Time'].values,
            kdata[['Time','LTS']].dropna()['Time'].values, kdata[['Time','LHO']].dropna()['Time'].values,
            idata[['Time','RTS']].dropna()['Time'].values, idata[['Time','RHO']].dropna()['Time'].values,
            idata[['Time','LTS']].dropna()['Time'].values, idata[['Time','LHO']].dropna()['Time'].values)



BW(kinematic_file = 'BW Kinematic data (example).csv',
       IMU_file = 'BW IMU data (example).csv')
