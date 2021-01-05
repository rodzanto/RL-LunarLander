#This is the reward function. You can edit this reward function to change the behavior of your lunar lander agent. This reward function is parameterized, with weighted coefficients C1 and C2. You can change the weights of these coefficients to optimize your agent. C1 corresponds to the coordinates of the lunar lander agent and C2 corresponds to the fuel used by the agent.

import numpy as np

def my_reward_function(state,m_power,s_power, prev_shaping):
	# Inputs: State vector, weighting coefficients, main and side power used
	# Outputs: reward value
    
    # ---------------------------------------------------------------------------

    # Most accurate landing
    c1 = 500   

    # Minimum fuel
    c2 = 0.5

    reward = 0
    
    #The following determines how fast are you moving, how far from center you are, the tilt angle, leg 1 in contact, leg 2 in contact with ground 
    shaping = - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
            - c1*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
            - 100*abs(state[4]) + 10*state[6] + 10*state[7]  

    # All components used in the shaping variable above are related to the physical 
    # states of the lander (position, velocity etc.). Ideally, these values are all
    # zero at the end of an episode. We want to measure the cumulative improvement
    # of these values; this is done using:
    if prev_shaping is not None:
        reward = shaping - prev_shaping
    
    prev_shaping = shaping
    
    # Add fuel burned by main and side engines

    reward -= m_power*c2  # less fuel spent is better, about -30 for heurisic landing
    reward -= s_power*(c2/10.0)
    

    return reward, prev_shaping