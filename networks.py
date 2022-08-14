import os
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.layers import Dense

class criticNetwork(keras.Model):
    def __init__(self,nactions,directory,fc1dims = 512,
                 fc2dims = 512,name = 'critic'):
        super(criticNetwork,self).__init__()
        self.fc1dims = fc1dims
        self.fc2dims = fc2dims
        self.modelName = name
        self.directory = directory
        # the NN is saved as .h5 file to be used later
        self.chkptFile = os.path.join(self.directory,self.modelName+'.h5')
        
        # creating the neural network note that all the layers are separate 
        # and not connected to each other
        self.fc1 = Dense(self.fc1dims, activation='relu')
        self.fc2 = Dense(self.fc2dims, activation='relu')
        self.q = Dense(1, activation=None)
    
    def call(self,state,action):
        """_summary_

        Args:
            state (numpy array): _description_
            action (numpy array): _description_

        Returns:
            q(s,a): this is the Q value of the state-action pair predicted by the NN
        """
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        q = self.q(action_value)
        return q

class actorNetwork(keras.Model):
    def __init__(self,nactions,directory,fc1dims = 512,
                 fc2dims = 512,name = 'actor'):
        super(actorNetwork,self).__init__()
        self.fc1dims = fc1dims
        self.fc2dims = fc2dims
        self.modelName = name
        self.directory = directory
        # the NN is saved as .h5 file to be used later
        self.chkptFile = os.path.join(self.directory,self.modelName+'.h5')
        
        # creating the neural network note that all the layers are separate 
        # and not connected to each other
        self.fc1 = Dense(self.fc1dims, activation='relu')
        self.fc2 = Dense(self.fc2dims, activation='relu')
        self.mu = Dense(1, activation='tanh')
    
    def call(self,state):
        """_summary_

        Args:
            state (tensorflow tensor): state of the current environment

        Returns:
            tensorflow tensor: action predicted by the NN 
        """
        action = self.mu(self.fc2(self.fc1(state)))
        return action
    