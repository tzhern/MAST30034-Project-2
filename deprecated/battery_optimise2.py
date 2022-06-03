import pandas as pd
import numpy as np

from pyomo.environ import *
from pyutilib.services import register_executable, registered_executable
register_executable(name='glpsol')

def battery_optimisation(df, initial_period=0, final_period=63456, include_revenue=True):
    """
    Determine the optimal charge and discharge behavior of a battery based 
    in Victoria. Assuming pure foresight of future spot prices over every 
    half-hour period to maximise the revenue.
    PS: Assuming no degradation to the battery over the timeline.
    ----------
    Parameters
    ----------
    df : Dataframe that contains the spot prices of each half-hour period of Victoria

    Returns
    ----------
    A dataframe that contains charge and discharge amount and battery's opening 
    capacity for each half-hour period.
    """
    MIN_BATTERY_CAPACITY = 0
    MAX_BATTERY_CAPACITY = 580
    MAX_BATTERY_IN = 135
    MAX_BATTERY_OUT = 150
    EFFICIENCY = 0.9
    MARGINAL_LOSS_FACTOR = 0.991

    battery = ConcreteModel()

    # defining components of the objective model
    # battery parameters
    battery.Period = Set(initialize=list(df.period), ordered=True)
    battery.Price = Param(initialize=list(df.spot_price))

    # battery varaibles
    battery.Capacity = Var(battery.Period, bounds=(MIN_BATTERY_CAPACITY, MAX_BATTERY_CAPACITY))
    battery.Battery_flow = Var(battery.Period, bounds=(-MAX_BATTERY_OUT, MAX_BATTERY_IN))

    # Set constraints for the battery
    def capacity_constraint(battery, i):
        # Assuming the battery is empty in the first period
        if i == battery.Period.first():
            return battery.Capacity[i] == MIN_BATTERY_CAPACITY
        else:
            return battery.Capacity[i] == (battery.Capacity[i-1] + (battery.Battery_flow[i-1]))

    # Make sure the battery discharge the amount it actually has
    def over_discharge(battery, i):
        return battery.Battery_flow[i] >= -battery.Capacity[i]
    
    # Make sure the battery does not over charge
    def over_charge(battery, i):
        return battery.Battery_flow[i] <= (MAX_BATTERY_CAPACITY - battery.Capacity[i])

    # Defining the battery objective
    def maximise_profit(battery):
        return sum(df.loc[i, 'spot_price'] * battery.Battery_flow[i] for i in battery.Period)

    # Set constraint and objective for the battery
    battery.capacity_state = Constraint(battery.Period, rule=capacity_constraint)
    battery.over_charge = Constraint(battery.Period, rule=over_charge)
    battery.over_discharge = Constraint(battery.Period, rule=over_discharge)
    battery.objective = Objective(rule=maximise_profit, sense=maximize)

    # Maximise the objective
    opt = SolverFactory('mosek')
    opt.solve(battery)

    # retrieve the range for looping
    period = range(battery.Period[initial_period + 1], battery.Period[final_period + 1] + 1)
    battery_flow = [(value(battery.Battery_flow[i])) for i in period]
    capacity = [value(battery.Capacity[i]) for i in period]
    spot_price = [battery.Price.extract_values()[None][i] for i in period]

    result = pd.DataFrame(dict(
        spot_price=spot_price,
        battery_flow=battery_flow,
        opening_capacity=capacity))
                                           
    # attach timestamp to the result dataframe
    result['datetime'] = df.time
    
    # calculate market dispatch
    result['market_dispatch'] = np.where(result.battery_flow > 0,
                                         -result.battery_flow / EFFICIENCY,
                                         -result.battery_flow * EFFICIENCY)
    
    # convert columns charge_power & discharge_power to power
    result['power'] = np.where((result.market_dispatch < 0), 
                                result.market_dispatch * 2, 
                                result.market_dispatch * 2 / EFFICIENCY)
    
    result = result[['datetime', 'spot_price', 'power', 'market_dispatch', 'opening_capacity']]
    
    # calculate revenue
    if include_revenue:
        result['revenue'] = np.where(result.market_dispatch < 0, 
                              result.market_dispatch * result.spot_price / MARGINAL_LOSS_FACTOR,
                              result.market_dispatch * result.spot_price * MARGINAL_LOSS_FACTOR)
    
    return result