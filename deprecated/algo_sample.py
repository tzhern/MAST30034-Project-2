import pandas as pd
import numpy as np
import statistics

CHARGE_EFFICIENCY = 90
DISCHARGE_EFFICIENCY = 90
MARGINAL_LOSS_FACTOR = 0.991
BATTERY_CAPACITY = 580
BATTERY_POWER = 300


def calc_market_dispatch(raw_power, discharge_efficiency=DISCHARGE_EFFICIENCY):
    
    raw_power = np.where(raw_power < 0,
                         raw_power/2,
                         raw_power/2 * discharge_efficiency/100)
    
    return np.round(raw_power)


def calc_market_revenue(market_dispatch, spot_price, marginal_loss_factor=MARGINAL_LOSS_FACTOR):
    market_revenue = np.where(market_dispatch < 0, 
                              market_dispatch * spot_price / marginal_loss_factor,
                              market_dispatch * spot_price * marginal_loss_factor)
    
    return np.round(market_revenue)


def calc_closing_capacity(market_dispatch, opening_capacity, battery_capacity=BATTERY_CAPACITY, charge_efficiency=CHARGE_EFFICIENCY, discharge_efficiency=DISCHARGE_EFFICIENCY):
    
    closing_capacity = np.where(market_dispatch < 0,
                                 max(0, min(battery_capacity, opening_capacity - market_dispatch * charge_efficiency / 100)),
                                 max(0, min(battery_capacity, opening_capacity - market_dispatch * 100 / charge_efficiency)))

    return np.round(closing_capacity)


# Algorithm 1
'''----------------------------------------------------------------------------------------------------------------------'''


def calc_a1_raw_power(spot_price, opening_capacity, charge_level=100, discharge_level=110, battery_power=BATTERY_POWER,
                      battery_capacity=BATTERY_CAPACITY, charge_efficiency=CHARGE_EFFICIENCY,
                      discharge_efficiency=DISCHARGE_EFFICIENCY):
    if (spot_price < charge_level):
        return int(-min(battery_power, (battery_capacity - opening_capacity)/(charge_efficiency/100)*2))
    elif (spot_price > discharge_level):
        return int(min(battery_power, (opening_capacity/(discharge_efficiency/100))*2))

    else:
        return 0


def compute_algo_1(spot_price):
    df = pd.DataFrame({'spot_price': spot_price})

    # create columns
    df['raw_power'] = 0
    df['market_dispatch'] = 0
    df['closing_capacity'] = 0

    # get column index
    columns = list(df.columns)
    spot_price_idx = columns.index('spot_price')
    raw_power_idx = columns.index('raw_power')
    market_dispatch_idx = columns.index('market_dispatch')
    closing_capacity_idx = columns.index('closing_capacity')

    np = df.to_numpy()
    for i in range(len(np)):
        np_curr = np[i]
        spot_price = np_curr[spot_price_idx]
        if i == 0:
            opening_capacity = 0
        else:
            opening_capacity = np[i-1][closing_capacity_idx]
        raw_power = calc_a1_raw_power(
            spot_price, opening_capacity)  # calc raw power
        market_dispatch = calc_market_dispatch(
            raw_power)  # calc market dispatch
        closing_capacity = calc_closing_capacity(
            market_dispatch, opening_capacity)  # calc closing capacity

        # read value into array
        np_curr[raw_power_idx] = raw_power
        np_curr[market_dispatch_idx] = market_dispatch
        np_curr[closing_capacity_idx] = closing_capacity

    # replace df columns with np
    df[:] = np
    
    # calc market revenue
    df['market_revenue'] = calc_market_revenue(df.market_dispatch, df.spot_price)
    
    return df


'''----------------------------------------------------------------------------------------------------------------------'''


# Algorithm 2
'''----------------------------------------------------------------------------------------------------------------------'''


def calc_a2_raw_power(behaviour, opening_capacity, battery_power=BATTERY_POWER, battery_capacity=BATTERY_CAPACITY,
                      charge_efficiency=CHARGE_EFFICIENCY, discharge_efficiency=DISCHARGE_EFFICIENCY):
    if behaviour == -1:
        return -min(battery_power, (battery_capacity - opening_capacity)/(charge_efficiency/100)*2)
    elif behaviour == 1:
        return min(battery_power, opening_capacity/(discharge_efficiency/100)*2)
    else:
        return 0


'''----------------------------------------------------------------------------------------------------------------------'''


# Algorithm 3
'''----------------------------------------------------------------------------------------------------------------------'''


def calc_a3_raw_power(forecast, opening_capacity, percentage, battery_power=BATTERY_POWER,
                      battery_capacity=BATTERY_CAPACITY, charge_efficiency=CHARGE_EFFICIENCY,
                      discharge_efficiency=DISCHARGE_EFFICIENCY):
    if (forecast == -1):
        return -min(battery_power * percentage, (battery_capacity - opening_capacity)/(charge_efficiency/100)*2)
    elif (forecast == 1):
        return min(battery_power * percentage, opening_capacity*2)
    else:
        return 0


def quantile_exc(ser, q):
    ser_sorted = ser.sort_values()
    rank = q * (len(ser) + 1) - 1
    assert rank > 0, 'quantile is too small'
    rank_l = int(rank)
    return ser_sorted.iat[rank_l] + (ser_sorted.iat[rank_l + 1] - ser_sorted.iat[rank_l]) * (rank - rank_l)


def std_mean(ser, num, direction):
    return np.mean(ser)+direction*num*np.std(ser)


def calc_forecast(spot_price, window=10, lower_pctl=0.25, upper_pctl=0.75, method=1, name='spot_price', show=False):
    """
    ----------
    Parameters
    ----------
    df         : dataframe
    window     : n of look-ahead day (default=10)
    lower_pctl : lower percentile (default=0.25)
    upper_pctl : upper percentile (default=0.75)
    method     : method of quantile. 1 = excel method, 2 = python method. (default = 1)
    name       : name of spot price column (default='spot_price')
    show       : show lower and upper column (default=False)

    Returns
    -------
    df         : dataframe
    """
    
    df = pd.DataFrame({'spot_price': spot_price}).reset_index()
    
    if method == 1:
        df['lower'] = df.spot_price[::-1].shift(1).rolling(window).apply(lambda x: quantile_exc(x, lower_pctl))
        df['upper'] = df.spot_price[::-1].shift(1).rolling(window).apply(lambda x: quantile_exc(x, upper_pctl))
    elif method == 2:
        df['lower'] = df.spot_price[::-1].shift(1).rolling(window).quantile(lower_pctl, interpolation='linear')
        df['upper'] = df.spot_price[::-1].shift(1).rolling(window).quantile(upper_pctl, interpolation='linear')
    elif method == 3:
        df['lower'] = df.spot_price[::-1].shift(1).rolling(window).apply(lambda x: std_mean(x, 1, -1))
        df['upper'] = df.spot_price[::-1].shift(1).rolling(window).apply(lambda x: std_mean(x, 1, +1))

    df['forecast'] = np.where(
        df['spot_price'] < df['lower'], -1, np.where(
            df['spot_price'] > df['upper'], 1, 0))
    
    if show:
        return df.forecast.to_numpy(), df.lower.to_numpy(), df.upper.to_numpy()

    return df.forecast.to_numpy()


def init_percentage(forecast):
    # set all percentage to 1 if predict a charge or discharge
    df = pd.DataFrame({'forecast': forecast}).reset_index()
    df['percentage'] = 1
    columns = list(df.columns)
    forecast_idx = columns.index('forecast')
    percentage_idx = columns.index('percentage')
    arr = df.to_numpy()
    for i in range(len(arr)):
        arr_curr = arr[i]
        forecast = arr_curr[forecast_idx]
        if forecast == 0:
            arr_curr[percentage_idx] = 0
    df[:] = arr
    return df.percentage.to_numpy()


def compute_algo_3(spot_price, method=1, window=10, lower_pctl=0.25, upper_pctl=0.75):
    df = pd.DataFrame({'spot_price': spot_price}).reset_index()

    # create columns
    df['raw_power'] = 0
    df['market_dispatch'] = 0
    df['closing_capacity'] = 0
    # calc forecast signal
    df['forecast'], df['lower'], df['upper'] = calc_forecast(df.spot_price, window=window, 
                                   method=method, lower_pctl=lower_pctl, 
                                   upper_pctl=upper_pctl, show=True)  
    df['percentage'] = init_percentage(df.forecast)
    # get column index
    columns = list(df.columns)
    spot_price_idx = columns.index('spot_price')
    raw_power_idx = columns.index('raw_power')
    market_dispatch_idx = columns.index('market_dispatch')
    closing_capacity_idx = columns.index('closing_capacity')
    forecast_idx = columns.index('forecast')
    percentage_idx = columns.index('percentage')

    arr = df.to_numpy()
    for i in range(len(arr)):

        arr_curr = arr[i]
        spot_price = arr_curr[spot_price_idx]
        if i == 0:
            opening_capacity = 0
        else:
            opening_capacity = arr[i-1][closing_capacity_idx]

        forecast = arr_curr[forecast_idx]
        if forecast == 0:
            arr_curr[percentage_idx] = 0
        percentage = arr_curr[percentage_idx]

        raw_power = calc_a3_raw_power(
            forecast, opening_capacity, percentage=percentage)  # calc raw power
        market_dispatch = calc_market_dispatch(
            raw_power)  # calc market dispatch
        closing_capacity = calc_closing_capacity(
            market_dispatch, opening_capacity)  # calc closing capacity

        # read value into array
        arr_curr[raw_power_idx] = raw_power
        arr_curr[market_dispatch_idx] = market_dispatch
        arr_curr[closing_capacity_idx] = closing_capacity

    # replace df columns with np array
    df[:] = arr
    
    # calc market revenue
    df['market_revenue'] = calc_market_revenue(df.market_dispatch, df.spot_price)

    return df


'''----------------------------------------------------------------------------------------------------------------------'''

# Checkpoint 3
'''----------------------------------------------------------------------------------------------------------------------'''


def calc_raw_power_c3(charge, discharge, opening_capacity, battery_power=BATTERY_POWER, battery_capacity=BATTERY_CAPACITY, diacharge_efficiency=DISCHARGE_EFFICIENCY):
    if charge > 0:
        return -min(battery_power * batery_power, (battery_capacity - opening_capacity) / (charge_efficiency/100)*2)
    elif discharge > 0:
        return min(battery_power * discharge, opening_capacity / (discharge_efficiency/100) * 2)
    return 0


'''----------------------------------------------------------------------------------------------------------------------'''
