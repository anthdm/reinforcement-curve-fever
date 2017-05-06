import pygame
from game import Game
import random
import numpy as np
from collections import deque
from conv_net import createGraph
import tensorflow as tf

ACTIONS = 3
INITIAL_ESPSILON = 1.0
FINAL_ESPSILON  = 0.5
OBSERVE = 500.
EXPLORE = 500. 
REPLAY_MEMORY = 50000
BATCH_SIZE = 32
LR = 1e-6
GAMMA = 0.99

def get_current_state(timestep):
    if timestep <= OBSERVE:
        return "observe"
    elif timestep > OBSERVE and timestep <= OBSERVE + EXPLORE:
        return "explore"
    else:
        return "train"

def trainNetwork(s, readout, h_fc1, sess):
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(LR).minimize(cost)

    game = Game(640, 480)

    D = deque()

    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0 = game.step(do_nothing)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded: ", checkpoint.model_checkpoint_path)
    else:
        print("No previous network weights found")

    log_file = open("logs/log", 'w', 0)

    t = 0
    epsilon = INITIAL_ESPSILON 
    while(1):
        readout_t = readout.eval(feed_dict = {s: [s_t]})[0]
        a_t = np.zeros(ACTIONS)
        action_index = 0
        if random.random() <= epsilon:
            print('This is a random action')
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

        if epsilon > FINAL_ESPSILON and t > OBSERVE:
            epsilon -= (INITIAL_ESPSILON - FINAL_ESPSILON) / EXPLORE

        x_t1, r_t = game.step(a_t)
        if r_t == -1:
            log = "timestep={} score={}\n".format(t, game.latest_score)
            log_file.write(log)

        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis = 2)

        D.append((s_t, a_t, r_t, s_t1))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE:
            minibatch = random.sample(D, BATCH_SIZE)

            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s: s_j1_batch})
            for i in range(0, len(minibatch)):
                y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            train_step.run(feed_dict = {
                y: y_batch,
                a: a_batch,
                s: s_j_batch
            })

        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/curve-fever-dqn', global_step = t)

        print("TIMESTEP {} | STATE {} | EPSILON {} | ACTION {} | REWARD {} | Q_MAX {}").format(
            t, get_current_state(t), epsilon, action_index, r_t, np.max(readout_t))

def train():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createGraph()
    trainNetwork(s, readout, h_fc1, sess)


if __name__ == "__main__":
    train()