import pandas as pd
import numpy as np
from algo_sample import *

def calc_forecast_tom_1(spot_price, small=[8, 0.1, 0.8], medium=[12, 0.25, 0.75], large=[24, 0.05, 0.9]):
    """
    Notes: Use 3 timeframes to vote for forecast
    ----------
    Parameters
    ----------
    spot_price  : Spot price series
    small       : timeframe 1 [window, lower percentile, upper percentile]
    medium      : timeframe 2 [window, lower percentile, upper percentile]
    large       : timeframe 3 [window, lower percentile, upper percentile]

    Returns
    -------
    forecast    : forecast series
    """
    assert (len(small) == 3) and (len(medium) == 3) and (
        len(large) == 3), "must be length 3"
    #assert (small > 0 and small < 1) and (medium > 0 and medium < 1) and (large > 0 and large < 1), "percentile must be 0 < x < 1"

    df = pd.DataFrame(spot_price, columns=['spot_price'])

    # Small time frame
    window_s = small[0]
    lower_pctl_s = small[1]
    upper_pctl_s = small[2]
    df['lower_s'] = df.spot_price[::-
                                  1].shift(1).rolling(window_s).quantile(lower_pctl_s)  # window=6, 0.10
    df['upper_s'] = df.spot_price[::-
                                  1].shift(1).rolling(window_s).quantile(upper_pctl_s)  # window=6, 0.90
    # for last window_s
    df['lower_s'][-window_s::] = df.spot_price[-window_s:                                               :].quantile(lower_pctl_s, interpolation='linear')
    df['upper_s'][-window_s::] = df.spot_price[-window_s:                                               :].quantile(upper_pctl_s, interpolation='linear')

    df['forecast_s'] = np.where((df.spot_price < df.lower_s), -1, np.where(
                                (df.spot_price > df.upper_s), 1, 0))

    # Medium timeframe
    window_m = medium[0]
    lower_pctl_m = medium[1]
    upper_pctl_m = medium[2]
    df['lower_m'] = df.spot_price[::-
                                  1].shift(1).rolling(window_m).quantile(lower_pctl_m)  # window=12, 0.20
    df['upper_m'] = df.spot_price[::-
                                  1].shift(1).rolling(window_m).quantile(upper_pctl_m)  # window=12, 0.80
    # for last window_m
    df['lower_m'][-window_m::] = df.spot_price[-window_m:                                               :].quantile(lower_pctl_m)
    df['upper_m'][-window_m::] = df.spot_price[-window_m:                                               :].quantile(upper_pctl_m)

    df['forecast_m'] = np.where((df.spot_price <= df.lower_m), -1, np.where(
                                (df.spot_price >= df.upper_m), 1, 0))

    # Large timeframe
    window_l = large[0]
    lower_pctl_l = large[1]
    upper_pctl_l = large[2]
    df['lower_l'] = df.spot_price[::-
                                  1].shift(1).rolling(window_l).quantile(lower_pctl_l)  # window=24, 0.05
    df['upper_l'] = df.spot_price[::-
                                  1].shift(1).rolling(window_l).quantile(upper_pctl_l)  # window=24, 0.95
    # for last window_l
    df['lower_l'][-window_l::] = df.spot_price[-window_l:                                               :].quantile(lower_pctl_l)
    df['upper_l'][-window_l::] = df.spot_price[-window_l:                                               :].quantile(upper_pctl_l)

    df['forecast_l'] = np.where((df.spot_price < df.lower_l), -1, np.where(
                                (df.spot_price > df.upper_l), 1, 0))

    # Vote
    df['forecast'] = np.sign(df.forecast_m + df.forecast_l + df.forecast_s)

    forecast = df.forecast

    return forecast.to_numpy()


def calc_forecast_tom_2(spot_price, short=[8, 0.1, 0.9], medium=[12, 0.35, 0.75], spike_window=5):

    df = pd.DataFrame(spot_price, columns=[
                      'spot_price']).reset_index(drop=True)

    # Short timeframe
    window_s = short[0]
    lower_pctl_s = short[1]
    upper_pctl_s = short[2]
    df['lower_s'] = df.spot_price[::-1].shift(1).rolling(
        window_s, min_periods=1).quantile(lower_pctl_s)  # window=6, 0.10
    df['upper_s'] = df.spot_price[::-1].shift(1).rolling(
        window_s, min_periods=1).quantile(upper_pctl_s)  # window=6, 0.90

    df['forecast_s'] = np.where((df.spot_price < df.lower_s), -1, np.where(
                                (df.spot_price > df.upper_s), 1, 0))

    # Medium timeframe
    window_m = medium[0]
    lower_pctl_m = medium[1]
    upper_pctl_m = medium[2]

    # window=12, 0.20
    df['lower_m'] = df.spot_price[::-1].shift(1).rolling(
        window_m, min_periods=1).quantile(lower_pctl_m, interpolation='linear')
    # window=12, 0.80
    df['upper_m'] = df.spot_price[::-1].shift(1).rolling(
        window_m, min_periods=1).quantile(upper_pctl_m, interpolation='linear')

    df['forecast_m'] = np.where((df.spot_price <= df.lower_m), -1, np.where(
                                (df.spot_price >= df.upper_m), 1,
                                0))

    # Idendity spikes

    shift_m1 = df.spot_price.shift(-1)
    shift_p1 = df.spot_price.shift(1)
    shift_p2 = df.spot_price.shift(2)

    df['spike1'] = np.where((shift_m1 <= shift_p1) & (shift_m1 < shift_p1.rolling(spike_window).quantile(0.15)) &
                            ((df.spot_price - shift_p1)/shift_p1 >= 0.135), 1, np.where(
                           (df.spot_price <= shift_p2) & (df.spot_price < shift_p2.rolling(spike_window).quantile(0.15)) &
                           ((shift_p1 - shift_p2)/shift_p2 >= 0.135), -1,
        0))

    df['spike2'] = np.where((df.spot_price < df.spot_price.rolling(spike_window, center=True).quantile(0.05)) &
                            (df.spot_price.rolling(spike_window, center=True).std()/df.spot_price > 0.3), -1, 0)

    # Vote
    df['forecast'] = np.sign(
        df.forecast_m + df.forecast_s + df.spike1 + df.spike2)

    forecast = df.forecast

    return forecast.to_numpy()


def filter_forecast(spot_price, forecast):
    """
    Notes: 
    1) Remove charge/discharge signals after/before the last discharge/ first charge signal
    2) Select n smallest/higest spot price to charge/discharge if n of same signals > m
    ----------
    Parameters
    ----------
    spot_price  : Spot price series
    forecast    : forecast series

    Returns
    -------
    forecast    : forecast series
    """
    df = pd.DataFrame(
        {'spot_price': spot_price, 'forecast': forecast}).reset_index(drop=True)

    dict_forecast = {'-1': [], '1': []}

    columns = list(df.columns)
    spot_price_idx = columns.index('spot_price')
    forecast_idx = columns.index('forecast')

    # change 1 to 0 if 1 appeared before the first charge signal appeared
    df.forecast[:df[df.forecast == -1].index[0]] = 0
    # change -1 to 0 after the last discharge signal appeared
    df.forecast[df[df.forecast == 1][::-1].index[0] + 1:] = 0

    # Convert to numpy array for faster speed
    arr = df[df.forecast != 0].to_numpy()
    
    for i in range(len(arr)):
        arr_curr = arr[i]
        spot_price_curr = arr_curr[spot_price_idx]
        forecast_curr = arr_curr[forecast_idx]
        if forecast_curr == -1:
            dict_forecast['1'] = []
            dict_forecast['-1'].append([i, spot_price_curr])
            if len(dict_forecast['-1']) > 5:
                max_idx = np.argmax([item[1] for item in dict_forecast['-1']])
                idx = dict_forecast['-1'][max_idx][0]
                arr[idx][forecast_idx] = 0
                dict_forecast['-1'].pop(max_idx)
        elif forecast_curr == 1:  # if forecast value if 1
            dict_forecast['-1'] = []  # clear -1 dict
            # add current row to dict
            dict_forecast['1'].append([i, spot_price_curr])
            if len(dict_forecast['1']) > 4:  # if exceeded
                # choose the lowest price row
                min_idx = np.argmin([item[1] for item in dict_forecast['1']])
                # get index of the lowest price row
                idx = dict_forecast['1'][min_idx][0]
                # set the lowest price row forecast value to 0
                arr[idx][forecast_idx] = 0
                # remove the lowest price row from dict
                dict_forecast['1'].pop(min_idx)

    # replace df columns with np array
    df[df.forecast != 0] = arr

    forecast = df.forecast

    return forecast.to_numpy()

def optimize_dispatch(spot_price, forecast, closing_capacity=False):
    """
    Notes: Maximise profit by dispatching the best combination of discharge and charge
    PS: The market dispatch for charge/discharge is after/before the efficiency rate
    ----------
    Parameters
    ----------
    spot_price       : Spot price series
    forecast         : forecast series
    closing_capacity : Boolean. If True then return closing capacity (default=False)

    Returns
    -------
    forecast    : forecast series
    """

    BATTERY_CAPACITY = 580
    MAX_DISCHARGE = 150
    MAX_CHARGE = -135
    
    capacity = 0

    df = pd.DataFrame({'spot_price': spot_price, 'forecast': forecast, 'market_dispatch': 0}).reset_index(drop=True)
    idx_spot_price = 0
    idx_forecast = 1
    idx_market_dispatch = 2

    
    dict_forecast = {'charge': [], 'discharge': []}
    # charge    : [index, spot_price, market_dispatch, left]
    # discharge : [index, spot_price, market_dispatch]
    idx_dict_spot_price = 1
    idx_dict_market_dispatch = 2
    idx_dict_left = 3
    
    # skip rows with 0 forecast for faster speed
    filtered = df[df.forecast!=0].reset_index()
    idx_last_discharge = filtered[filtered.forecast == 1].tail(1).index[0]
    arr = df[df.forecast!=0].to_numpy()
    for i in range(len(arr)):
        arr_curr = arr[i]
        
        # if charge signal
        if arr_curr[idx_forecast] < 0:
            
            # add charge position to dict
            opening_capacity = capacity
            capacity = min(BATTERY_CAPACITY, capacity - MAX_CHARGE)
            capacity = max(0, capacity)
            market_dispatch = opening_capacity - capacity
            
            if market_dispatch == MAX_CHARGE:
                # if current capacity is not full, charge regularly
                dict_forecast['charge'].append([i, arr_curr[idx_spot_price], MAX_CHARGE, MAX_CHARGE])
            else:
                # if current capacity is full,
                # check if current price is lower than previous one and the previous one has leftover
                # if yes, we should undo the previous charge and charge here instead
                dict_forecast['charge'].sort(key=lambda x: x[idx_dict_spot_price], reverse=True)
                for charge in dict_forecast['charge']:
                    if arr_curr[idx_spot_price] < charge[idx_dict_spot_price]:
                        transferable = charge[idx_dict_left]
                        transfer = max(transferable, MAX_CHARGE - market_dispatch)
                        market_dispatch += transfer
                        charge[idx_dict_left] -= transfer
                        charge[idx_dict_market_dispatch] -= transfer
                        idx = charge[0]
                    else: 
                        break
                
                # transfer completed, remove charge from dict if charge is empty
                charge_position_new = []
                for charge in dict_forecast['charge']:
                    if charge[idx_dict_left] < 0:
                        charge_position_new.append(charge)
                dict_forecast['charge'] = charge_position_new
                dict_forecast['charge'].append([i, arr_curr[idx_spot_price], market_dispatch, market_dispatch])

            
                
        # if discharge signal
        elif arr_curr[idx_forecast] > 0:
            dict_forecast['discharge'].append([i, arr_curr[idx_spot_price], MAX_DISCHARGE])
            arr_curr[idx_market_dispatch] = MAX_DISCHARGE
            
            # we discharge at the end of each discharge sequences
            try:
                isNextCharge = arr[i+1][idx_forecast] < 0
            except:
                isNextCharge = True
                
            if isNextCharge:
                
                # we need to sort charge and discharge position so
                # we can sell the lowest charge at the higest price
                dict_forecast['charge'].sort(key=lambda x: x[idx_dict_spot_price])
                dict_forecast['discharge'].sort(key=lambda x: x[idx_dict_spot_price], reverse=True)
                    
                # sell discharge position if available
                for discharge in dict_forecast['discharge']:
                    for charge in dict_forecast['charge']:
                        discharge[idx_dict_market_dispatch] += charge[idx_dict_left]
                        capacity += charge[idx_dict_left]

                        # regular charge
                        if discharge[idx_dict_market_dispatch] >= 0:
                            charge[idx_dict_left] = 0

                        # if it is fully discharged and have remaining charge,
                        # give charge back to dict_forecast_charge
                        else:
                            charge[idx_dict_left] = discharge[idx_dict_market_dispatch]
                            discharge[idx_dict_market_dispatch] = 0
                            capacity -= charge[idx_dict_left]
                            
                # remove empty charge position
                charge_position_new = []
                for charge in dict_forecast['charge']:
                    left = charge[idx_dict_left]
                    used = charge[idx_dict_market_dispatch] - left
                    idx = charge[0]
                    arr[idx][idx_market_dispatch] = used
                    if left < 0:
                        charge_position_new.append(charge)
                dict_forecast['charge'] = charge_position_new
                
                # we adjust the charge amount at the end so we wont have
                # any capacity after the last discharge completed
                if i == len(arr) - 1:
                    for charge in dict_forecast['charge']:
                        idx = charge[0]
                        arr[idx][idx_market_dispatch] = charge[idx_dict_market_dispatch] - charge[idx_dict_left]
                
                for discharge in dict_forecast['discharge']:
                    idx = discharge[0]
                    if discharge[idx_dict_market_dispatch] == MAX_DISCHARGE:
                        # set discharge forcast to zero if 
                        # the current discharge position did not discharge at all
                        arr[idx][idx_market_dispatch] = 0
                    else:
                        # set the discharge amount (market dispatch)
                        arr[idx][idx_market_dispatch] = min(MAX_DISCHARGE, MAX_DISCHARGE - discharge[idx_dict_market_dispatch])
                    
                # discharge complete, remove discharge from dict
                # and wait for the next discharge cycle
                dict_forecast['discharge'] = []
            
    # replace df columns with np array
    df[df.forecast != 0] = arr

    # take charge/discharge efficiency into account
    df['market_dispatch'] = np.where(df.market_dispatch < 0, 
                                     df.market_dispatch/0.9,
                                     df.market_dispatch*0.9)
    
    if closing_capacity:
        df['closing_capacity'] = np.where(df.market_dispatch < 0, 
                                          -df.market_dispatch*0.9,
                                          -df.market_dispatch/0.9)
        df['closing_capacity'] = df.closing_capacity.cumsum()
        
        return df.market_dispatch.to_numpy(), df.closing_capacity.to_numpy()
    
    return df.market_dispatch.to_numpy()
