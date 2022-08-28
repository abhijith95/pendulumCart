import numpy as np
import math
from pendulumCart import pendulumCart
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from agent import agent

agent = agent(inputDims = 5, actionBound=50)
actor = keras.models.Sequential()
actor.add(agent.actor.fc1)
actor.add(agent.actor.fc2)
actor.add(agent.actor.mu)
temp = actor(tf.convert_to_tensor([np.zeros(5)], dtype=tf.float32))
# actor.load_weights(r'C:\Users\abhij\pendulumCart\NN_weights\actor.h5')
actor = keras.models.load_model(r'C:\Users\abhij\Desktop\my_model')

def testModel():
    episodeTime = 25 
    FREQ = 1/0.02
    fig,axs = plt.subplots(3,1,sharex=True)
    axs[0].set_title("Pendulum position vs time")
    axs[0].set_ylim([-180,180])
    axs[1].set_title("Cart position vs time")
    axs[2].set_title("Force vs time")
    axs[2].set_xlabel("Time")
    axs[0].set_ylabel("Angular position (deg)")
    axs[1].set_ylabel("Cart position (m)")
    axs[2].set_ylabel("Velocity(m/s)")
    
    simulationTime = 0
    initAngle = 180
    envTest = pendulumCart(initialStateVector=[0,0,initAngle*math.pi/180,0],
                           timeStep=1/FREQ)
    time,pendulumAngle,cartPos,force = [0],[initAngle],[0],[0]
    while simulationTime < episodeTime:
        state = envTest.getState()
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = actor(state)
        cartForce = float(action)
        envTest.systemSolver(simulationTime,cartForce,True)
        simulationTime+=(1/FREQ)
        temp = math.degrees(envTest.pendulumPosition)
        # if temp < 0:
        #     temp+=360
        # if 359 < temp < 361:
        #     temp = 0
        time.append(simulationTime)
        pendulumAngle.append(temp)
        cartPos.append(envTest.cartPosition)
        force.append(cartForce)
        
    axs[0].plot(time,pendulumAngle)
    axs[1].plot(time,cartPos)
    axs[2].plot(time,force)
    plt.show()

testModel()