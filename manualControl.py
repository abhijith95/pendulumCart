import numpy as np
import math
from pendulumCart import pendulumCart
from Controllers.pidController import pidController
import matplotlib.pyplot as plt

DT = 1/1000

class manualControl(pendulumCart,pidController):
    """
    This is a trial class to see if the system is controllable
    This has a simple PID controller that attempts to bring the pendulum to the
    top and make sure it stays there.
    """
    def __init__(self):
        pendulumCart.__init__(self,cartMass=0.5, pendulumMass=0.2,pendulumLength=0.3,
                               damping = 0.1, springStiffness=0,timeStep=DT,
                               initialStateVector=[0,0,5*math.pi/180,0])
        pidController.__init__(self,kp = 10,ki = 0.10, kd = 1, dt = DT)
        self.currentTime = 0
        self.targetPosition = 0
        
        # Initializing graphs and setting the axes titles
        self.fig,self.axs = plt.subplots(3,1,sharex=True)
        self.axs[0].set_title("Pendulum position vs time")
        self.axs[1].set_title("Cart position vs time")
        self.axs[2].set_title("Force vs time")
        self.axs[2].set_xlabel("Time")
        self.axs[0].set_ylabel("Angular position (deg)")
        self.axs[1].set_ylabel("Cart position (m)")
        self.axs[2].set_ylabel("Force(N)")
    
    def systemUpdate(self):
        actuatorSignal = self.computeOutput(self.targetPosition,self.pendulumPosition)
        self.systemSolver(self.currentTime,actuatorSignal)
        return actuatorSignal
    
    def plotGraph(self,time,pendulumAngle,cartPos,force):
        self.axs[0].plot(time,pendulumAngle)
        self.axs[1].plot(time,cartPos)
        self.axs[2].plot(time,force)
        plt.show()
    
    def runSystem(self):
        time,pendulumAngle,cartPos,force = [0],[180],[0],[0]
        while self.currentTime < 3:
            f = self.systemUpdate()
            temp = math.degrees(self.pendulumPosition)
            if temp < 0:
                temp+=360
            if 359 < temp < 361:
                temp = 0
            time.append(self.currentTime)
            pendulumAngle.append(temp)
            cartPos.append(self.cartPosition)
            force.append(f)
            # self.plotGraph()
            # self.fig.canvas.draw()
            self.currentTime+=DT
        self.plotGraph(time,pendulumAngle,cartPos,force)
        
mc = manualControl()
mc.runSystem()
print("Done")