from types import new_class
import numpy as np

class memoryBuffer:
    """
    This class is used to define the memory buffer where
    the experiences from the agent are stored, to be used
    to train the NN later on. Numpy array is used in this 
    example to store the necessary details, but also check
    out "deque" in python to store the replays 
    """
    def __init__(self,maxSize,stateShape,nactions):
        self.memorySize = maxSize
        self.memoryCtr = 0 # to keep track of how many entries are filled in the buffer
        self.stateMemory = np.zeros((maxSize,stateShape))
        self.newStateMemory = np.zeros((maxSize,stateShape))
        self.actionMemory = np.zeros((maxSize,nactions))
        self.rewardMemory = np.zeros((maxSize))
        self.terminalMemory = np.zeros(maxSize,dtype=np.bool)
    
    def storeTransition(self,state,action,reward,newState,done):
        # done is the flag that says if the episode is over or not
        index = self.memoryCtr % self.memorySize # this gives the index of the first available memory
        self.stateMemory[index] = state
        self.newStateMemory[index] = newState
        self.rewardMemory[index] = reward
        self.actionMemory[index] = action
        self.terminalMemory[index] = 1-done
        self.memoryCtr+=1
        
    def sampleBuffer(self,batchSize):
        # function that returns a minibatch from the memory
        maxmem = min(self.memoryCtr,self.memorySize) # to see if the memory is filled or not
        batch = np.random.choice(maxmem,batchSize,replace=False)
        
        state = self.stateMemory[batch]
        newState = self.newStateMemory[batch]
        reward = self.rewardMemory[batch]
        action = self.actionMemory[batch]
        done = self.terminalMemory[batch]
        
        return state,action,reward,newState,done