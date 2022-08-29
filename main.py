import numpy as np
import math
from agent import agent
from pendulumCart import pendulumCart
import tensorflow as tf
import matplotlib.pyplot as plt

FREQ = 1/0.02
pcAgent = agent(inputDims = 5, actionBound=50)
nGames = 1000
episodeTime = 25 # in sec
env = pendulumCart(timeStep=1/FREQ)
scoreHistory = []

bestScore = -np.inf
startAngle = []
endAngle = []
for i in range(nGames):
    env.reset() # resetting the environment before each episode
    done = False
    score = 0
    simulationTime = 0
    startAngle.append(math.degrees(env.pendulumPosition))
    print("Training episode",i+1)
    while not done:
        state = env.getState()
        action = pcAgent.takeAction(state)
        # solve the ODE to move the system to next state
        # action = float(action)*env.maxForce
        env.systemSolver(simulationTime,float(action),True)
        simulationTime+=(1/FREQ)
        newState = env.getState()
        reward = env.getReward(action)
        score+=reward
        done = env.isDone()
        done = done or (simulationTime > episodeTime)
        pcAgent.remember(state,action,reward,newState,done)
        pcAgent.learn()
        
    endAngle.append(math.degrees(env.pendulumPosition))    
    scoreHistory.append(score)
    if score > bestScore:
        bestScore = score
        pcAgent.saveModels()
        bestModel = pcAgent.actor
        bestModel.save(r'C:\Users\abhij\pendulumCart\NN_weights\bestActor')

fig,axs = plt.subplots(3,1,sharex=True)  
axs[0].plot(range(nGames),startAngle)
axs[1].plot(range(nGames),endAngle)
axs[2].plot(range(nGames),scoreHistory)
plt.show()
#%%
def testModel(initAngle):
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
    envTest = pendulumCart(initialStateVector=[0,0,initAngle*math.pi/180,0],
                           timeStep=1/FREQ)
    time,pendulumAngle,cartPos,force = [0],[initAngle],[0],[0]
    while simulationTime < episodeTime:
        state = envTest.getState()
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = bestModel(state)
        cartForce = float(action[0])
        envTest.systemSolver(simulationTime,action,True)
        simulationTime+=(1/FREQ)
        temp = math.degrees(envTest.pendulumPosition)
        # if temp < 0:
        #     temp+=360
        # if 359 < temp < 361:
        #     temp = 0
        time.append(simulationTime)
        pendulumAngle.append(temp)
        cartPos.append(envTest.cartPosition)
        force.append(envTest.cartVelocity)
        
    axs[0].plot(time,pendulumAngle)
    axs[1].plot(time,cartPos)
    axs[2].plot(time,force)
    plt.show()

testModel(200)
print("Done")