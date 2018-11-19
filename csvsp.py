## csv signal processing
# 相関関数から信号の遅延を推定(正の相関のみに対応)
# 相関関数から相関係数を推定
import pandas
import numpy as np

""" ----------------------------------------------------------------------------
## read csv data
# @param filepath
# @param meas_header
# @param ref_header
# @param invalid_values
# @return _data
# @return _data_meas
# @return _data_ref
---------------------------------------------------------------------------- """
def _read_data(filepath, meas_header, ref_header, invalid_values=[]):
    try:
        _data = pandas.read_csv(filepath)
        _data_meas = _data[meas_header]
        _data_ref = _data[ref_header]
        for invalid_value in invalid_values:
            _data_meas[_data_meas==invalid_value] = np.nan
            _data_ref[_data_ref==invalid_value] = np.nan
        _data_meas = _data_meas.interpolate(limit_direction="backward").dropna()
        _data_ref = _data_ref.interpolate(limit_direction="backward").dropna()
        return _data, _data_meas, _data_ref
    except Exception as e:
        print("Error:csvsp._read_data : %s"%e.args[0])
        return np.nan

""" ----------------------------------------------------------------------------
## calculate Correlation Function
# @param filepath
# @param meas_header
# @param ref_header
# @param invalid_values
# @return _corrfunc
---------------------------------------------------------------------------- """
def corrfunc(filepath, meas_header, ref_header, invalid_values=[]):
    try:
        _data, _data_meas, _data_ref = _read_data(filepath, meas_header, ref_header, invalid_values)
        _corrfunc = np.correlate(_data_meas-np.nanmean(_data_meas),
                                 _data_ref-np.nanmean(_data_ref),
                                 "full")
        return _corrfunc
    except Exception as e:
        print("Error:csvsp.coeffunc : %s"%e.args[0])
        return np.nan

""" ----------------------------------------------------------------------------
## calculate measured signal delay
# @param filepath
# @param meas_header
# @param ref_header
# @param invalid_values
# @return _delay
---------------------------------------------------------------------------- """
def delay(filepath, meas_header, ref_header, invalid_values=[]):
    try:
        _data, _data_meas, _data_ref = _read_data(filepath, meas_header, ref_header, invalid_values)
        _corrfunc = np.correlate(_data_meas-np.nanmean(_data_meas),
                                 _data_ref-np.nanmean(_data_ref),
                                 "full")
        _delay = _corrfunc.argmax() - (len(_data_ref)-1)
        return _delay
    except Exception as e:
        print("Error:csvsp.delay : %s"%e.args[0])
        return np.nan

""" ----------------------------------------------------------------------------
## align measured signal and reference signal
# @param filepath
# @param meas_header
# @param ref_header
# @param invalid_values
# @return _data_concat pandas.DataFrame
---------------------------------------------------------------------------- """
def align(filepath, meas_header, ref_header, invalid_values=[]):
    try:
        _data, _data_meas, _data_ref = _read_data(filepath, meas_header, ref_header, invalid_values)
        _corrfunc = np.correlate(_data_meas-np.nanmean(_data_meas),
                                 _data_ref-np.nanmean(_data_ref),
                                 "full")
        _delay = _corrfunc.argmax() - (len(_data_ref)-1)
        if _delay < 0:
            _data_meas = _data[meas_header]
            _data_ref = _data[ref_header].drop(range(_delay)).reset_index(drop=True)
        elif _delay > 0:
            _data_meas = _data[meas_header].drop(range(abs(_delay))).reset_index(drop=True)
            _data_ref = _data[ref_header]
        else:
            _data_meas = _data[meas_header]
            _data_ref = _data[ref_header]
        _data_concat = pandas.concat([_data_meas,_data_ref],1).dropna()
        return _data_concat
    except Exception as e:
        print("Error:csvsp.align : %s"%e.args[0])
        return np.nan

""" ----------------------------------------------------------------------------
## calculate correlaton coefficient
# @param filepath
# @param meas_header
# @param ref_header
# @param invalid_values
# @return _corrcoef
---------------------------------------------------------------------------- """
def corrcoef(filepath, meas_header, ref_header, invalid_values=[]):
    try:
        _data_concat = align(filepath, meas_header, ref_header, invalid_values=[])
        _corrcoef = np.corrcoef(_data_concat[meas_header], _data_concat[ref_header])
        return _corrcoef
    except Exception as e:
        print("Error:csvsp.coefcorr : %s"%e.args[0])
        return np.nan
