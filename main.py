import numpy as np
import math
from agent import agent
from pendulumCart import pendulumCart
import tensorflow as tf
import matplotlib.pyplot as plt

pcAgent = agent(5)
nGames = 500
env = pendulumCart()
scoreHistory = []

bestScore = -np.inf
for i in range(nGames):
    env.reset() # resetting the environment before each episode
    done = False
    score = 0
    simulationTime = 0
    while not done:
        state = env.getState()
        action = pcAgent.takeAction(state)
        # solve the ODE to move the system to next state
        # action = float(action)*env.maxForce
        env.systemSolver(simulationTime,action)
        simulationTime+=1
        newState = env.getState()
        reward = env.getReward(action)
        score+=reward
        done = env.isDone()
        done = done or (simulationTime > 2500)
        pcAgent.remember(state,action,reward,newState,done)
        pcAgent.learn()
        
    scoreHistory.append(score)
    if score > bestScore:
        bestScore = score
        pcAgent.saveModels()
        bestModel = pcAgent.actor
    
plt.plot(range(nGames),scoreHistory)
plt.show()
def testModel():
    fig,axs = plt.subplots(3,1,sharex=True)
    axs[0].set_title("Pendulum position vs time")
    axs[1].set_title("Cart position vs time")
    axs[2].set_title("Force vs time")
    axs[2].set_xlabel("Time")
    axs[0].set_ylabel("Angular position (deg)")
    axs[1].set_ylabel("Cart position (m)")
    axs[2].set_ylabel("Force(N)")
    
    simulationTime = 0
    envTest = pendulumCart(initialStateVector=[0,0,math.pi,0])
    time,pendulumAngle,cartPos,force = [0],[180],[0],[0]
    while simulationTime < 2500:
        state = envTest.getState()
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = bestModel(state)
        cartForce = float(action)*envTest.maxForce
        envTest.systemSolver(simulationTime,action)
        simulationTime+=1
        temp = math.degrees(envTest.pendulumPosition)
        if temp < 0:
            temp+=360
        if 359 < temp < 361:
            temp = 0
        time.append(simulationTime)
        pendulumAngle.append(temp)
        cartPos.append(envTest.cartPosition)
        force.append(cartForce)
        
    axs[0].plot(time,pendulumAngle)
    axs[1].plot(time,cartPos)
    axs[2].plot(time,force)
    plt.show()

# testModel()
print("Done")