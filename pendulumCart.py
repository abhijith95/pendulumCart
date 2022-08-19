import numpy as np
import math
import random
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
    def __init__(self,cartMass=1, pendulumMass=1,pendulumLength=0.1,
                               damping = 0.1, springStiffness=0,timeStep=1/1000,
                               initialStateVector=[0,0,5*math.pi/180,0],
                 maxTrackLength = 1,maxForce = 20):
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
        self.friction = -(pendulumMass+cartMass)*G
        self.M = pendulumMass + cartMass
        self.maxTrackLength = maxTrackLength
        self.maxForce = maxForce
    
    def getState(self):
        state = np.array([math.sin(self.pendulumPosition),math.cos(self.pendulumPosition),
                          self.pendulumVelocity,self.cartPosition,
                          self.cartVelocity], dtype=np.float32)
        return state
    
    def isDone(self):
        # the episode is finished if the cart goes off track or time for the episode exceeds 25sec
        if abs(self.cartPosition) > self.maxTrackLength:
            return 1
        else:
            return 0
    
    def reset(self):
        self.cartPosition = random.uniform(0,self.maxTrackLength)
        self.cartVelocity,self.pendulumVelocity = 0.0, 0.0
        temp = random.uniform(0,360)
        self.pendulumPosition = math.radians(temp)
    
    def systemEquation(self,t,y,f):
        """
        Equations to the system can be found here:
        https://www.12000.org/my_notes/cart_motion/report.htm#x1-20001
        f is the total external force including any disturbances
        Note here that theta is measured from the vertical and in clockwise direction
        Theta of 0 is actually equal to theta of pi/2 according to regular convention
        x1,x2,x3,x4 = x,xdot,theta,thetadot
        """
        k = 1/3
        x1,x2,x3,x4 = y
        stheta,ctheta = math.sin(x3),math.cos(x3)
        commonTerm = f + (self.damping*self.friction*math.tanh(x2)) + (0.5*self.pendulumMass*self.pendulumLength*math.sin(x3)*(x4**2))
        # commonTerm = f + (self.damping*self.friction*math.tanh(x2)) + (self.pendulumMass*self.pendulumLength*stheta*(x4**2))
        x1dot = x2
        x2dot = (commonTerm - (0.75*self.pendulumMass*G*math.sin(x3)*math.cos(x3)))/(self.pendulumMass+self.cartMass-(0.75*self.pendulumMass*( (math.cos(x3))**2)))
        # x2dot = (self.pendulumMass*G*stheta*ctheta - (1+k)*commonTerm)/(self.pendulumMass*(ctheta**2) - (1+k)*self.M)
        x3dot = x4
        x4dot = ((1.5*G*math.sin(x3)/self.pendulumLength) - (1.5*math.cos(x3)*commonTerm/(self.pendulumLength*(self.cartMass+self.pendulumMass))))/(1-(0.75*((math.cos(x3))**2)/(self.cartMass+self.pendulumMass)))
        # x4dot = ((self.M*G*stheta) - ctheta*commonTerm)/((1+k)*self.M*self.pendulumLength - (self.pendulumMass*self.pendulumLength*(ctheta**2)))
        dydt = [x1dot,x2dot,x3dot,x4dot]
        return dydt
    
    def systemSolver(self,currentTime,systemInput):
        y0 = [self.cartPosition,self.cartVelocity,self.pendulumPosition,self.pendulumVelocity]
        solution = solve_ivp(fun = self.systemEquation,
                             t_span =[currentTime,currentTime+self.timeStep],t_eval = [currentTime+self.timeStep],
                             y0=y0,args=[systemInput])
        self.cartPosition,self.cartVelocity,self.pendulumPosition,self.pendulumVelocity = float(solution.y[0]),float(solution.y[1]),float(solution.y[2]),float(solution.y[3])
        self.pendulumPosition = self.pendulumPosition % (2*math.pi)
    
    def getReward(self,systemInput):
        """
        The reward function is setup in such a way that following parameters are penalized:
            1. pendulum position - goal is to get the pendlumum to zero angle
            2. Cart force - not to spend too much effort in moving the cart
        This function returns reward for the current state of the system
        The parameters are taken from the paper: https://www.mdpi.com/2076-3417/10/24/9013
        """    
        ar,br,cr,dr,er,n = (10**-2),(0.1),(0),(-0.01),(-100),2
        r1 = dr*(br*(abs(self.pendulumPosition*180/math.pi)**n)+cr*(abs(systemInput)**n))
        if abs(self.cartPosition) < self.maxTrackLength:
            reward = r1
        else:
            reward = r1+er
        return reward
        

# pcSystem = pendulumCart(cartMass=10,pendulumMass=1,
#                         pendulumLength=0.5,damping=1,springStiffness=10,
#                         timeStep=10,initialStateVector=[0,0,math.pi/2,0])
# pcSystem.systemUpdate(1,100.0)
        