import numpy as np
from pendulumCart import pendulumCart

class agent(pendulumCart):
    """
    The agent class contains the necessary code implementation
    to train the agent. Since the action is in a continuous space
    we need actor critic model to solve the control problem of
    the inverted pendulum on cart model
    
    The goal is to swing up the pendulum and balance it up, just
    by moving the cart.
    """