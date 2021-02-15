# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 23:51:23 2021

@author: Benst
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import StockEnv
import time
import pandas

gamma = 0.99
eps = np.finfo(np.float32).eps.item()

num_batch = 1
num_inputs = 5
num_actions = 3
num_hidden = 128

inputs = layers.Input(batch_shape = (1,num_batch,num_inputs))
#common = layers.Dense(100, activation="relu")(inputs)
lstm1 = layers.LSTM(50, return_sequences = True, activation="tanh", stateful = True)(inputs)
drop1 = layers.Dropout(0.2)(lstm1)
lstm2 = layers.LSTM(50, return_sequences = True, activation = 'relu', stateful = True)(drop1)
drop2 = layers.Dropout(0.2)(lstm2)
lstm3 = layers.LSTM(units = 25, return_sequences = False, activation = 'relu', stateful = True)(drop2)
action = layers.Dense(num_actions, activation="softmax")(lstm3)
critic = layers.Dense(1)(lstm3)

model = keras.Model(inputs=inputs, outputs=[action, critic])

optimizer = keras.optimizers.Adam(lr=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []

env = StockEnv.StockEnv("KO", 1000,num_batch, 0 )

while True:  # Run until solved
    state = env.State.to_array()
    with tf.GradientTape() as tape: 
        for timestep in range(1, 10):
            x = tf.Variable([[state]], trainable=True, dtype=tf.float32)
            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(x)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            #print(action)
            action_probs_history.append(tf.math.log(action_probs[0, action]))
            
            if(np.isnan(action_probs).any()):
                action = 0
                print("NAN")
            else:                
                action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            # Apply the sampled action in our environment
            state, reward = env.Step(action)
            rewards_history.append(reward)
            
            template = "Balance: {:.3f}  Shares: {:.1f}  Price:{:.3f}  Action: {}  Reward: {:.3f}  Value:{:.3f}  Time: {}"
            
            print(template.format(env.State.Balance * env.BalScale, env.State.Shares * env.ShareScale, 
                                  env.State.Price* env.PriceScale, env.LastAction, reward*env.RewardScale, 
                                  env.GetStateValue(env.State), env.TimeStep))


        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, rewards_history)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()







