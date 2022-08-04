import pyglet
from pyglet.window import key
import numpy as np
import math
from pendulumCart import pendulumCart
from Controllers.pidController import pidController

DT = 1
class render(pyglet.window.Window,pendulumCart):
    
    def __init__(self,*args,**kwargs):
        
        pyglet.window.Window.__init__(self,*args,**kwargs)
        pendulumCart.__init__(self,cartMass=5, pendulumMass=1,pendulumLength=0.5,
                               damping = 0.1, springStiffness=2,timeStep=DT,
                               initialStateVector=[0,0,math.pi,0])
        self.controller = pidController(kp = 10,ki = 0, kd = 1, dt = DT)
        
        self.scale = 5.0
        self.yPos = 100
        self.currentPos = np.array([100,180]) # the SW corner of the cart and angular position of the pendulum
        self.cartWidth, self.cartHeight = 80,50
        self.pendulumLength, self.pendulumWidth = 70,10
        self.pivotRadius, self.wheelRadius = 2 , 8
        self.currentTime = 0
        self.targetPosition = 0
    
    def on_draw(self):
        self.clear()
        # drawing cart
        R = self.pendulumWidth/math.sqrt(2)
        cartCenter = [self.currentPos[0] + (0.5*self.cartWidth),
                      self.yPos + (0.5*self.cartHeight)]
        cart = pyglet.shapes.Rectangle(x = self.currentPos[0], y = self.yPos,
                                       width= self.cartWidth, height=self.cartHeight,
                                       color = (0,255,0))
        
        # drawing the pendulum
        """
        Since the pendulum swings, the SW corner point of the pendulum changes with the rotation.
        THis is what is depicted in the coordinates below.
        """
        pendulum = pyglet.shapes.Rectangle(x = cartCenter[0] - (R*math.sin(math.radians(45+self.currentPos[1]))),
                                           y = cartCenter[1] - (R*math.cos(math.radians(45+self.currentPos[1]))),
                                           width = self.pendulumWidth, height= self.pendulumLength,
                                           color = (200,0,0))
        pendulum.rotation=self.currentPos[1]
        
        # drawing the pivot pin
        pin = pyglet.shapes.Circle(x = cartCenter[0],y = cartCenter[1],
                                   radius = self.pivotRadius)
        
        # drawing wheels for cart
        rearWheel = pyglet.shapes.Circle(x = self.currentPos[0]+self.wheelRadius,
                                         y = self.yPos-self.wheelRadius,
                                         radius=self.wheelRadius)
        frontWheel = pyglet.shapes.Circle(x = self.currentPos[0]-self.wheelRadius+self.cartWidth,
                                         y = self.yPos-self.wheelRadius,
                                         radius=self.wheelRadius)
        
        # drawing all the shapes
        cart.draw()
        pendulum.draw()
        pin.draw()
        rearWheel.draw()
        frontWheel.draw()
    
    def update(self,dt):
        # pass
        actuatorSignal = self.controller.computeOutput(self.targetPosition,self.pendulumPosition)
        self.systemSolver(self.currentTime,actuatorSignal)
        pendulumAngle = math.degrees(self.pendulumPosition)
        if pendulumAngle < 0:
            pendulumAngle+=360
        self.currentPos= self.currentPos + np.array([self.cartPosition*self.scale,pendulumAngle])
        self.currentTime+=dt

gameWindow = render(600,600,caption = "Pendulum on a cart",resizable = True)
pyglet.clock.schedule_interval(gameWindow.update, DT)
pyglet.app.run() # command to execute running the window