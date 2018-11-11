## csv signal processing
# 相関関数から信号の遅延を推定(正の相関のみに対応)
# 相関関数から相関係数を推定

import pandas
import numpy as np
import matplotlib.pyplot as plt

# testdata
t = np.arange(0,1,0.001)
theta0 = 0
theta1 = 0.1
sin0 = np.sin(2*np.pi*t+theta0)
sin1 = np.sin(2*np.pi*t+theta1)
signal = pandas.DataFrame([sin0,sin1],index=["are","sore"]).T
plt.plot(signal["are"])
plt.plot(signal["sore"])
plt.show()

# test variable
filepath = "testdata0.csv"
meas_header = "are"
ref_header = "sore"


def corrfunc(filepath, meas_header, ref_header):
    try:
        _data = pandas.read_csv(filepath)
        _corrfunc = np.correlate(_data[meas_header]-np.nanmean(_data[meas_header]),
                                 _data[ref_header]-np.nanmean(_data[ref_header]),
                                 "full")
        return _corrfunc
    except Exception as e:
        print("Error:csvsp.coeffunc : %s"%e.args[0])
        return np.nan

def delay(filepath, meas_header, ref_header):
    try:
        _data = pandas.read_csv(filepath)
        _corrfunc = np.correlate(_data[meas_header]-np.nanmean(_data[meas_header]),
                                 _data[ref_header]-np.nanmean(_data[ref_header]),
                                 "full")
        _delay = _corrfunc.argmax() - (len(_data[ref_header])-1)
        return _delay
    except Exception as e:
        print("Error:csvsp.delay : %s"%e.args[0])
        return np.nan

def coefcorr(filepath, meas_header, ref_header):
    try:
        _data = pandas.read_csv(filepath)
        _corrfunc = np.correlate(_data[meas_header]-np.nanmean(_data[meas_header]),
                                 _data[ref_header]-np.nanmean(_data[ref_header]),
                                 "full")
        _corrcoef = max(_corrfunc)
        return _corrcoef
    except Exception as e:
        print("Error:csvsp.coefcorr : %s"%e.args[0])
        return np.nan

def align(filepath, meas_header, ref_header):
    try:
        _data = pandas.read_csv(filepath)
        _corrfunc = np.correlate(_data[meas_header]-np.nanmean(_data[meas_header]),
                                 _data[ref_header]-np.nanmean(_data[ref_header]),
                                 "full")
        _delay = _corrfunc.argmax() - (len(_data[ref_header])-1)
        if _delay < 0:
            _meas_data = _data[meas_header]
            _ref_data = _data[ref_header].drop(range(_delay)).reset_index(drop=True)
        elif _delay > 0:
            _meas_data = _data[meas_header].drop(range(abs(_delay))).reset_index(drop=True)
            _ref_data = _data[ref_header]
        _data_concat = pandas.concat([_meas_data,_ref_data],1).dropna()
        return _data_concat
    except Exception as e:
        print("Error:csvsp.align : %s"%e.args[0])
        return np.nan
