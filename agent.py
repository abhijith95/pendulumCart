import numpy as np
from pendulumCart import pendulumCart
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.keras.optimizers import Adam
from memoryBuffer import memoryBuffer
from networks import criticNetwork, actorNetwork

class agent:
    """
    The agent class contains the necessary code implementation
    to train the agent. Since the action is in a continuous space
    we need actor critic model to solve the control problem of
    the inverted pendulum on cart model
    
    The goal is to swing up the pendulum and balance it up, just
    by moving the cart.
    
    Follow this paper on agent training: https://arxiv.org/abs/1509.02971
    """
    def __init__(self,inputDims,alpha = 0.001,beta = 0.002,
                 gamma = 0.99,nactions = 1, maxMemorySize = 10**6,
                 tau = 0.005,fc1 = 128, fc2 = 200,batchSize = 128, noise = 0.1,
                 actionBound = 20):
        """_summary_

        Args:
            inputDims (_type_): _description_
            alpha (float, optional): Learning rate for actor network. Defaults to 0.001.
            beta (float, optional): Learning rate for critic network. Defaults to 0.002.
            gamma (float, optional): discount factor for the future rewards. Defaults to 0.9.
            nactions (int, optional): number of possible actions taken by the agent. Defaults to 1.
            maxMemorySize (int, optional): max size of memory buffer. Defaults to 10000.
            tau (float, optional): the update of target network weights. Defaults to 0.005.
            fc1 (int, optional): number of nuerons in the first hidden layer. Defaults to 128.
            fc2 (int, optional): number of neurons in the second hidden layer. Defaults to 200.
            batchSize (int, optional): sample size of memory to train the networks. Defaults to 64.
        """
        self.gamma = gamma
        self.tau = tau
        self.memory = memoryBuffer(maxMemorySize,inputDims,nactions)
        self.batchSize = batchSize
        self.nactions = nactions
        self.noise = noise
        self.env = pendulumCart()
        self.actionBound = actionBound
        
        self.actor = actorNetwork(nactions,r'C:\Users\abhij\pendulumCart\NN_weights')
        self.critic = criticNetwork(r'C:\Users\abhij\pendulumCart\NN_weights')
        self.targetActor = actorNetwork(nactions,r'C:\Users\abhij\pendulumCart\NN_weights',
                                        name = 'target-actor')
        self.targetCritic = criticNetwork(r'C:\Users\abhij\pendulumCart\NN_weights',
                                          name = 'target-critic')
        
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.targetActor.compile(optimizer=Adam(learning_rate=alpha))
        self.targetCritic.compile(optimizer=Adam(learning_rate=beta))

        self.updateNetworkParams(tau=1)
    
    def updateNetworkParams(self,tau=None):
        if tau is None:
            tau = self.tau
        
        weights = []
        targets = self.targetActor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.targetActor.set_weights(weights)
        
        weights = []
        targets = self.targetCritic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.targetCritic.set_weights(weights)
    
    def remember(self,state,action,reward,newState,done):
        self.memory.storeTransition(state,action,reward,newState,done)
    
    def saveModels(self):
        print('.......saving models.....')
        self.actor.save_weights(self.actor.chkptFile)
        self.targetActor.save_weights(self.targetActor.chkptFile)
        self.critic.save_weights(self.critic.chkptFile)
        self.targetCritic.save_weights(self.targetCritic.chkptFile)
    
    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.chkptFile)
        self.targetActor.load_weights(self.targetActor.chkptFile)
        self.critic.load_weights(self.critic.chkptFile)
        self.targetCritic.load_weights(self.targetCritic.chkptFile)
    
    def takeAction(self,state,evaluate = False):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = tf.multiply(self.actionBound,self.actor(state))
        if not evaluate:
            actions += tf.random.normal(shape=[self.nactions],
                                        mean=0.0, stddev=self.noise)
        actions = tf.clip_by_value(actions, -self.actionBound, self.actionBound)
        return actions[0]
    
    def learn(self):
        if self.memory.memoryCtr < self.batchSize:
            return

        state, action, reward, newState, done = \
            self.memory.sampleBuffer(self.batchSize)
        
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(newState, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            target_actions = tf.multiply(self.actionBound,self.targetActor(states_))
            critic_value_ = tf.squeeze(self.targetCritic(
                                states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma*critic_value_*(1-done)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))
        
        with tf.GradientTape() as tape:
            new_policy_actions = tf.multiply(self.actionBound,self.actor(states))
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))
        
        self.updateNetworkParams()