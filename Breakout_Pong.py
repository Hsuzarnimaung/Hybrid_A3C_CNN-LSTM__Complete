import gym

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools

import threading
import multiprocessing
import pandas as pd
from nets import create_networks
from worker import Worker
from datetime import datetime
start=datetime.now()

ENV_NAME ="SpaceInvaders-v4"
MAX_GLOBAL_STEPS = 1e4
STEPS_PER_UPDATE = 5

def Env(ENV_NAME):
    return gym.envs.make(ENV_NAME)

env = Env(ENV_NAME)
NUM_ACTIONS = env.action_space.n
env.close()

def smooth(x):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:(i+1)].sum() / (i - start + 1))
    return y

# Set number of workers
NUM_WORKERS =4 # This one run with 12 processors
tf.compat.v1.reset_default_graph()
with tf.device("/cpu:0"):
    # Keeps track of number of updates we've performed
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Global policy and value nets
    with tf.variable_scope("global") as vs:
        net = create_networks(NUM_ACTIONS)

        # Global step iterator
    global_counter = itertools.count()
    global_counter2 = itertools.count()
    #epsodie_count=itertools.count()

    # Save returns
    returns_list = []
    step_list=[]
    step_Reward = []
    Advantage = []
    V_S = []
    Q_S = []
    V_Loss = []
    P_Loss = []
    action_count = []
    A_A=[]
    step_count=[]
    workers_name=[]
    # Create workers
    workers = []
    for worker_id in range(NUM_WORKERS-1):
        worker = Worker(
            name="worker_{}_{}".format(ENV_NAME,worker_id),
            env=Env(ENV_NAME),
            net=net,
            name2=ENV_NAME,
            step_list=step_list,
            global_counter=global_counter,
            returns_list=returns_list,
            discount_factor = 0.99,
            max_global_steps=MAX_GLOBAL_STEPS,
            step_Reward=step_Reward,
            Advantage=Advantage,
            V_S=V_S,
            Q_S=Q_S,
            P_Loss=P_Loss,
            V_Loss=V_Loss,
            action_count=action_count,A_A=A_A,step_count=step_count,workers_name=workers_name
            )
        workers.append(worker)
    worker = Worker(
        name="worker_{}_{}".format(ENV_NAME, NUM_WORKERS-1),
        env=gym.envs.make(ENV_NAME),
        net=net,
        name2=ENV_NAME,
        step_list=step_list,
        global_counter=global_counter,
        returns_list=returns_list,
        discount_factor=0.99,
        max_global_steps=MAX_GLOBAL_STEPS,
        step_Reward=step_Reward,
        Advantage=Advantage,
        V_S=V_S,
        Q_S=Q_S,
        P_Loss=P_Loss,
        V_Loss=V_Loss,
        action_count=action_count,A_A=A_A,step_count=step_count,workers_name=workers_name
    )
    workers.append(worker)

    returns_list2=[]
    step_list2=[]
    step_Reward2 = []
    Advantage2 = []
    V_S2 = []
    Q_S2 = []
    V_Loss2 = []
    P_Loss2 = []
    action_count2 = []
    A_A2=[]
    step_count2=[]
    workers_name2=[]
    ENV_NAME="DemonAttack-v4"
    for worker_id in range(NUM_WORKERS-1):
        worker = Worker(
            name="worker_{}_{}".format(ENV_NAME,worker_id),
            env=Env(ENV_NAME),
            net=net,
            name2=ENV_NAME,
            step_list=step_list2,
            global_counter=global_counter2,
            returns_list=returns_list2,
            discount_factor=0.99,
            render_mode=False,
            max_global_steps=MAX_GLOBAL_STEPS,
            step_Reward=step_Reward2,
            Advantage=Advantage2,
            V_S=V_S2,
            Q_S=Q_S2,
            P_Loss=P_Loss2,
            V_Loss=V_Loss2,
            action_count=action_count2,A_A=A_A2,step_count=step_count2,workers_name=workers_name2
        )
        workers.append(worker)
    worker = Worker(
        name="worker_{}_{}".format(ENV_NAME, NUM_WORKERS-1),
        env=gym.envs.make(ENV_NAME),
        net=net,
        name2=ENV_NAME,
        step_list=step_list2,
        global_counter=global_counter2,
        returns_list=returns_list2,
        discount_factor=0.99,
        render_mode=True,
        max_global_steps=MAX_GLOBAL_STEPS,
        step_Reward=step_Reward2,
        Advantage=Advantage2,
        V_S=V_S2,
        Q_S=Q_S2,
        P_Loss=P_Loss2,
        V_Loss=V_Loss2,
        action_count=action_count2,A_A=A_A2,step_count=step_count2,workers_name=workers_name2
    )
    workers.append(worker)
    saver = tf.train.Saver(max_to_keep=STEPS_PER_UPDATE)
with tf.Session() as sess:
    if (False):
        print('Loading Model of Breakout and Pong...')
        ckpt = tf.train.get_checkpoint_state("Model/Breakout_Pong")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()

    # Start worker threads
    worker_threads = []
    for worker in workers:
        worker_fn = lambda: worker.run(sess, coord, STEPS_PER_UPDATE)
        t = threading.Thread(target=worker_fn)
        t.start()
        worker_threads.append(t)

    # wait for all workers to finish
    coord.join(worker_threads, stop_grace_period_secs=300)
    saver.save(sess, "./Model/RiverRaid_Adventure_After" + "/model" + str("_global_step") + ".cptk")
    end = datetime.now()
    print(f"Training time:{end - start}")
    df = pd.DataFrame.from_dict({"Rewards": returns_list,"Global_Step":step_list,"Predict_total_Reward":step_Reward,
                                 "Choosed_Action_Count":action_count,"Average Advantage":A_A,"Average Value Loss":V_Loss,"Total_loss":V_S,
                                 "Average Policy Loss":P_Loss,"Length of episode":step_count,"List of workers_name":workers_name})
    df.to_excel("./Excel/Hybrid_Total_SpaceInvaders_CNN.xlsx", index=False)
    df = pd.DataFrame.from_dict({"Rewards": returns_list2,"Global_Step":step_list2,"Predict_total_Reward":step_Reward2,
                                 "Choosed_Action_Count":action_count2,"Average Advantage":A_A2,"Average Value Loss":V_Loss2,
                                 "Average Policy Loss":P_Loss2,"Length of episode":step_count2,"List of workers_name":workers_name2})
    df.to_excel("./Excel/Hybrid_Total_DemonAttack_CNN.xlsx", index=False)
    # Plot smoothed returns
    x = np.array(returns_list)
    y = smooth(x)
    plt.title("Space Invaders")
    plt.plot(x, label='orig')
    plt.plot(y, label='smoothed')
    plt.legend()
    plt.show()
    x = np.array(returns_list2)
    y = smooth(x)
    plt.title("Demon Attack")
    plt.plot(x, label='orig')
    plt.plot(y, label='smoothed')
    plt.legend()
    plt.show()