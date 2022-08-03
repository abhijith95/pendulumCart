import pyglet
from pyglet.window import key
import numpy as np
import math
from pendulumCart import pendulumCart
from Controllers import pidController

class render(pyglet.window.Window):
    
    def __init__(self,*args,**kwargs):
        
        pyglet.window.Window.__init__(self,*args,**kwargs)
        # pendulumCart.__init__(self,cartMass=5, pendulumMass=1,pendulumLength=0.5,
        #                        damping = 0.1, springStiffness=2,timeStep=1,
        #                        initialStateVector=[0,0,math.pi,0])
        # pidController.__init(self,kp = 10,ki = 0, kd = 1)
        
        self.scale = 5
        self.currentPos = np.array([100,100,90]) # the SW corner of the cart and angular position of the pendulum
        self.cartWidth, self.cartHeight = 80,50
        self.pendulumLength, self.pendulumWidth = 70,10
        self.pivotRadius, self.wheelRadius = 2 , 8
        self.currentTime = 0
        self.targetPosition = 0
    
    def on_draw(self):
        self.clear()
        # drawing cart
        cartCenter = [self.currentPos[0] + (0.5*self.cartWidth),
                      self.currentPos[1] + (0.5*self.cartHeight)]
        cart = pyglet.shapes.Rectangle(x = self.currentPos[0], y = self.currentPos[1],
                                       width= self.cartWidth, height=self.cartHeight,
                                       color = (0,255,0))
        
        # drawing the pendulum
        pendulum = pyglet.shapes.Rectangle(x = cartCenter[0] - (0.5*self.pendulumWidth),
                                           y = cartCenter[1] - (0.5*self.pendulumWidth),
                                           width = self.pendulumWidth, height= self.pendulumLength,
                                           color = (200,0,0))
        # pendulum.anchor_x,pendulum.anchor_y = cartCenter[0]-pendulum.x,cartCenter[1]-pendulum.y
        # pendulum.anchor_x,pendulum.anchor_y = 5,5
        pendulum.rotation=self.currentPos[2]
        
        # drawing the pivot pin
        pin = pyglet.shapes.Circle(x = cartCenter[0],y = cartCenter[1],
                                   radius = self.pivotRadius)
        
        # drawing wheels for cart
        rearWheel = pyglet.shapes.Circle(x = self.currentPos[0]+self.wheelRadius,
                                         y = self.currentPos[1]-self.wheelRadius,
                                         radius=self.wheelRadius)
        frontWheel = pyglet.shapes.Circle(x = self.currentPos[0]-self.wheelRadius+self.cartWidth,
                                         y = self.currentPos[1]-self.wheelRadius,
                                         radius=self.wheelRadius)
        
        # drawing all the shapes
        cart.draw()
        pendulum.draw()
        pin.draw()
        rearWheel.draw()
        frontWheel.draw()
    
    def update(self,dt):
        pass
        # self.currentPos+=cartPosition*self.scale
        # actuatorSignal = self.computeOutput(self.targetPosition,self.pendulumPosition)
        # self.systemSolver(self.currentTime,actuatorSignal)

gameWindow = render(600,600,caption = "Pendulum on a cart",resizable = True)
pyglet.clock.schedule_interval(gameWindow.update, 1)
pyglet.app.run() # command to execute running the window