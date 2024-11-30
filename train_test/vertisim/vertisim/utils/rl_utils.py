"""
Utility functions for reinforcement learning
"""

from .units import sec_to_min

def charge_reward(current_soc, soc_reward_threshold, current_demand):
    # Rewards the charging action if the current state of charge is below the threshold
    if current_demand == 0:
        return round(10-(current_soc/100), 2)
    else:
        if current_soc <= soc_reward_threshold:
            return round(1-(current_soc/100), 2)
        else:
            return -1      

def get_flight_duration_estimation(aircraft, system_manager):
    destination_vertiport_id = system_manager.get_random_destination(aircraft=aircraft)

    flight_duration_estimation = sec_to_min(system_manager.get_average_flight_time(
        flight_direction=f"{aircraft.current_vertiport_id}_{destination_vertiport_id}"))
    
    return flight_duration_estimation
