import numpy as np
import math
from scipy.integrate import solve_ivp

G = 9.81
class pendulumCart:
    """
    This class defines the physics model of pendulum on a cart.
    The class has the properties like mass of the cart, pendulum,
    also keeps track of current state space vectors. The input
    to the system will be passed onto the function  whose output
    will be the update in the new position and velocity.
    """
    def __init__(self,cartMass,pendulumMass,pendulumLength,
                 damping,springStiffness,timeStep,initialStateVector):
        """
        Defining the basic properties of the system
        """
        self.cartMass = cartMass
        self.pendulumMass = pendulumMass
        self.pendulumLength = pendulumLength
        self.damping,self.springStiffness = damping,springStiffness
        self.cartPosition,self.cartVelocity,self.pendulumPosition,self.pendulumVelocity = initialStateVector
        self.externalForce = 0
        self.timeStep = timeStep
    
    def systemEquation(self,t,y,f):
        """
        Equations to the system can be found here:
        https://www.12000.org/my_notes/cart_motion/report.htm#x1-20001
        f is the total external force including any disturbances
        Note here that theta is measured from the vertical and in clockwise direction
        Theta of 0 is actually equal to theta of pi/2 according to regular convention
        """
        x1,x2,x3,x4 = y
        commonTerm = f - (self.damping*x2) - (self.springStiffness*x1) + (0.5*self.pendulumMass*self.pendulumLength*math.sin(x3)*(x4**2))
        x1dot = x2
        x2dot = (commonTerm - (0.75*self.pendulumMass*G*math.sin(x3)*math.cos(x3)))/(self.pendulumMass+self.cartMass-(0.75*self.pendulumMass*( (math.cos(x3))**2)))
        x3dot = x4
        x4dot = ((1.5*G*math.sin(x3)/self.pendulumLength) - (1.5*math.cos(x3)*commonTerm/(self.pendulumLength*(self.cartMass+self.pendulumMass))))/(1-(0.75*((math.cos(x3))**2)/(self.cartMass+self.pendulumMass)))
        dydt = [x1dot,x2dot,x3dot,x4dot]
        return dydt
    
    def systemUpdate(self,currentTime,f):
        y0 = [self.cartPosition,self.cartVelocity,self.pendulumPosition,self.pendulumVelocity]
        solution = solve_ivp(fun = self.systemEquation,
                             t_span =[currentTime,currentTime+self.timeStep],t_eval = [currentTime+self.timeStep],
                             y0=y0,args=[f])
        self.cartPosition,self.cartVelocity,self.pendulumPosition,self.pendulumVelocity = solution.y

# pcSystem = pendulumCart(cartMass=10,pendulumMass=1,
#                         pendulumLength=0.5,damping=1,springStiffness=10,
#                         timeStep=10,initialStateVector=[0,0,math.pi/2,0])
# pcSystem.systemUpdate(1,100.0)
        