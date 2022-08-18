import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, BatchNormalization

class criticNetwork(keras.Model):
    def __init__(self,directory,fc1dims = 400,
                 fc2dims = 300,name = 'critic'):
        super(criticNetwork,self).__init__()
        self.fc1dims = fc1dims
        self.fc2dims = fc2dims
        self.modelName = name
        self.directory = directory
        # the NN is saved as .h5 file to be used later
        self.chkptFile = os.path.join(self.directory,self.modelName+'.h5')
        self.buildNetwork()
    
    def buildNetwork(self):        
        # creating the neural network note that all the layers are separate 
        # and not connected to each other
        # self.fc1 = Dense(self.fc1dims, activation='relu')
        # self.fc2 = Dense(self.fc2dims, activation='relu')
        # self.q = Dense(1, activation=None)
        inputLayer = keras.layers.Input(shape=(5))
        dense1 = Dense(self.fc1dims)(inputLayer)
        batch1 = BatchNormalization(dense1)
        self.fc1 = tf.nn.relu(batch1)
        # self.fc1 = Dense(self.fc1dims)
        dense2 = Dense(self.fc2dims)(self.fc1)
        batch2 = BatchNormalization(dense2)
        actionIn = Dense(self.fc2dims,activation = 'relu')(self.actions)
        self.fc2 = tf.nn.relu(tf.add(batch2,actionIn))
        # self.fc2 = Dense(self.fc2dims, activation='relu')
        self.q = Dense(1, activation=None)(self.fc2) 
        self.model = keras.models.Model(inputs = inputLayer, oututs = self.q)
    
    def call(self,state,action):
        """_summary_

        Args:
            state (numpy array): _description_
            action (numpy array): _description_

        Returns:
            q(s,a): this is the Q value of the state-action pair predicted by the NN
        """
        self.actions = action
        # action_value = self.fc1(tf.concat([state, action], axis=1))
        # action_value = self.fc2(action_value)
        q = self.model(state)
        return q

class actorNetwork(keras.Model):
    def __init__(self,nactions,directory,fc1dims = 400,
                 fc2dims = 300,name = 'actor',actionBounds = 20):
        super(actorNetwork,self).__init__()
        self.fc1dims = fc1dims
        self.fc2dims = fc2dims
        self.modelName = name
        self.directory = directory
        # the NN is saved as .h5 file to be used later
        self.chkptFile = os.path.join(self.directory,self.modelName+'.h5')
        self.bound = actionBounds
        self.nactions = nactions
        self.buildNetwork()
    
    def buildNetwork(self):
        # creating the neural network note that all the layers are separate 
        # and not connected to each other
        inputLayer = keras.layers.Input(shape=(5))
        dense1 = Dense(self.fc1dims)(inputLayer)
        batch1 = BatchNormalization(dense1)
        self.fc1 = tf.nn.relu(batch1)
        # self.fc1 = Dense(self.fc1dims)
        dense2 = Dense(self.fc2dims)(self.fc1)
        batch2 = BatchNormalization(dense2)
        self.fc2 = tf.nn.relu(batch2)
        # self.fc2 = Dense(self.fc2dims, activation='relu')
        self.mu = Dense(self.nactions, activation='tanh')(self.fc2)
        self.model = keras.models.Model(inputs = inputLayer, oututs = self.mu)
    
    def call(self,state):
        """_summary_

        Args:
            state (tensorflow tensor): state of the current environment

        Returns:
            tensorflow tensor: action predicted by the NN 
        """
        action = self.model(state) * self.bound
        return action
    