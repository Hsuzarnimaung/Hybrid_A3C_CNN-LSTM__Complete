import os.path

import numpy as np
import tensorflow as tf

from nets import create_networks
import cv2
from array2gif import write_gif
from PIL import Image
from PIL import Image
import numpy as np
class Step:
  def __init__(self, state, action, reward, next_state, expected_value,done):
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state
    self.expected_value=expected_value
    self.done = done


# Transform raw images for input into neural network
# 1) Convert to grayscale
# 2) Resize
# 3) Crop
class ImageTransformer:
  def __init__(self,name):
    with tf.variable_scope("image_transformer"):
      if(name=="Adventure-v4"):
        self.input_state = tf.placeholder(shape=[250, 160, 3], dtype=tf.uint8)
      else: self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
      self.output = tf.image.rgb_to_grayscale(self.input_state)
      self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
      self.output = tf.image.resize(self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      self.output = tf.squeeze(self.output)
      #self.output= tf.to_float(self.output) / 255.0
      #self.output=tf.to_float(self.output)

  def transform(self, state, sess=None):
    sess = sess or tf.get_default_session()
    return sess.run(self.output, { self.input_state: state })


# Create initial state by repeating the same frame 4 times
def repeat_frame(frame):
  return np.stack([frame] * 4, axis=2)


# Create next state by shifting each frame by 1
# Throw out the oldest frame
# And concatenate the newest frame
def shift_frames(state, next_frame):
  return np.append(state[:,:,1:], np.expand_dims(next_frame, 2), axis=2)


# Make a Tensorflow op to copy weights from one scope to another
def get_copy_params_op(src_vars, dst_vars):
  src_vars = list(sorted(src_vars, key=lambda v: v.name))
  dst_vars = list(sorted(dst_vars, key=lambda v: v.name))

  ops = []
  for s, d in zip(src_vars, dst_vars):
    op = d.assign(s)
    ops.append(op)

  return ops


def make_train_op(local_net, global_net):
  """
  Use gradients from local network to update the global network
  """
  p_local_grads, _ = zip(*local_net.grads_and_vars)
  p_local_grads, _ = tf.clip_by_global_norm(p_local_grads, 40.0)
  _, p_global_vars = zip(*global_net.grads_and_vars)
  local_grads_global_vars = list(zip(p_local_grads, p_global_vars))

  local_grads, _ = zip(*local_net.v_grads_and_vars)
  local_grads, _ = tf.clip_by_global_norm(local_grads, 40.0)
  _, global_vars = zip(*global_net.v_grads_and_vars)
  v_local_grads_global_vars = list(zip(local_grads, global_vars))

  return global_net.optimizer.apply_gradients(
    local_grads_global_vars,
    global_step=tf.train.get_global_step()),global_net.voptimizer.apply_gradients(
    v_local_grads_global_vars,
    global_step=tf.train.get_global_step())






# Worker object to be run in a thread
# name (String) should be unique for each thread
# env (OpenAI Gym Environment) should be unique for each thread
# policy_net (PolicyNetwork) should be a global passed to every worker
# value_net (ValueNetwork) should be a global passed to every worker
# returns_list (List) should be a global passed to every worker
class Worker:
  def __init__(
      self,
      name,
      env,
      net,
      name2,
      global_counter,
      returns_list,
      step_list,
      step_Reward,
      Advantage,
          V_S,
          Q_S,
          V_Loss,
          P_Loss,
          action_count,
          step_count,

          A_A,
          workers_name,
      discount_factor=0.99,
      max_global_steps=None,
      render_mode=False,):

    self.name = name
    self.env = env
    self.global_net = net
    self.name2=name2
    self.global_counter = global_counter
    self.discount_factor = discount_factor
    self.max_global_steps = max_global_steps
    self.global_step = tf.train.get_global_step()
    self.img_transformer = ImageTransformer(self.name2)
    self.step_list=step_list
    self.render_mode=render_mode

    self.step_Reward = step_Reward
    self.Advantage = Advantage
    self.V_S = V_S
    self.Q_S = Q_S
    self.V_Loss = V_Loss
    self.P_Loss = P_Loss
    self.action_count=action_count
    self.step_count=step_count
    self.episode_Count=0
    self.A_A=A_A
    self.workers_name=workers_name
    # Create local policy and value networks that belong only to this worker
    with tf.variable_scope(name):

      self.net = create_networks(net.num_outputs)

    # We will use this op to copy the global network weights
    # back to the local policy and value networks
    self.copy_params_op = get_copy_params_op(
      tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="global"),
      tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name))

    # These will take the gradients from the local networks
    # and use those gradients to update the global network
    self.net_train_op = make_train_op(self.net, self.global_net)
    self.length=0.
    self.predict_reward=0.
    self.action_c=np.zeros((self.net.num_outputs), np.int64)
    self.A_P_l=0.
    self.A_V_l=0.
    self.A_ad=0.
    self.state = None # Keep track of the current state
    self.total_reward = 0. # After each episode print the total (sum of) reward
    self.returns_list = returns_list # Global returns list to plot later
    self.e_c=0.
    self.d_state=None
    if not os.path.exists("Image/" + self.name):
      os.mkdir("Image/" + self.name)


    # ... get array s.t. arr.shape = (3,256, 256)




  def run(self, sess, coord, t_max):
    with sess.as_default(), sess.graph.as_default():
      # Assign the initial state
      self.state = repeat_frame(self.img_transformer.transform(self.env.reset()))
      cv2.imwrite("Image/" + self.name + "/" + str(self.e_c) + "/" + "Initial.png", self.env.reset())
      self.d_state=repeat_frame(self.env.reset())

      if not os.path.exists("Image/"+self.name+"/"+str(self.e_c)):
        os.mkdir("Image/"+self.name+"/"+str(self.e_c))
      if not os.path.exists("Image/"+self.name+"/"+str(self.e_c)+"test"):
        os.mkdir("Image/"+self.name+"/"+str(self.e_c)+"test")
      try:
        while not coord.should_stop():
          # Copy weights from  global networks to local networks
          sess.run(self.copy_params_op)

          # Collect some experience
          steps, global_step = self.run_n_steps(t_max, sess)

          # Stop once the max number of global steps has been reached
          if self.max_global_steps is not None and global_step >= self.max_global_steps:
            coord.request_stop()
            return

          # Update the global networks using local gradients
          policy_loss,value_loss,advantage,action,value_target=self.update(steps, sess)
          self.A_P_l+=policy_loss
          self.A_V_l+=value_loss


          if(steps[-1].done):
            self.net.h=tf.zeros((1,256))
            self.net.c=tf.zeros((1,256))
            self.step_list.append(global_step)
            self.P_Loss.append(self.A_P_l / self.length)
            self.V_Loss.append(self.A_V_l / self.length)
            self.V_S.append(self.A_V_l)

            self.e_c+=1

            self.episode_Count = 0.
            self.A_V_l = 0.
            self.A_P_l = 0.
            self.A_ad=0.
            self.length = 0.
            if not os.path.exists("Image/"+self.name+"/" + str(self.e_c)):
                os.mkdir("Image/" +self.name+"/" +str(self.e_c))
            if not os.path.exists("Image/"+self.name+"/" + str(self.e_c)+"test"):
                os.mkdir("Image/" +self.name+"/" +str(self.e_c)+"test")



      except tf.errors.CancelledError:
        return

  def sample_action(self, state, sess):
    # Make input N x D (N = 1)
    feed_dict = { self.net.states: [state] }
    actions ,expected_value= sess.run([self.net.sample_action,self.net.vhat], feed_dict)

    # Prediction is a 1-D array of length N, just want the first value
    return actions[0],expected_value[0]


  def run_n_steps(self, n, sess):
    steps = []
    for _ in range(n):


      # Take a step
      action,value = self.sample_action(self.state, sess)
      next_frame, reward, done, _ = self.env.step(action)
    #   after 100 episodes start to render worker
      self.action_c[action]+=1

      # Shift the state to include the latest frame
      next_state = shift_frames(self.state, self.img_transformer.transform(next_frame))
      next_s=shift_frames(self.d_state, next_frame)
      #print(self.d_state)
      cv2.imwrite("Image/" + self.name + "/" + str(self.e_c) + "/"  + str(self.length) + ".png", self.state)
      cv2.imwrite("Image/" + self.name + "/" + str(self.e_c) + "test/" + str(self.length) + ".png", next_frame)

      with open("Image/" + self.name + "/" + str(self.e_c) + ".txt", "a") as f:
        #for word in words:
            f.write(str(action)+" "+str(reward)+" "+str(value)+" ")
            f.write("\n")
      self.length+=1
      self.predict_reward+=value
      # Save total return
      if done:
        print("Total reward:", self.total_reward, "Worker:", self.name,"Length:",self.length)
        self.returns_list.append(self.total_reward)
        self.step_count.append(self.length)
        self.step_Reward.append(self.predict_reward)
        self.action_count.append(self.action_c)
        self.workers_name.append(self.name)
        self.A_A.append(self.A_ad / self.length)

        if len(self.returns_list) > 0 and len(self.returns_list) % 100 == 0:
          print("*** Total average reward (last 100):", np.mean(self.returns_list[-100:]), "Collected so far:", len(self.returns_list),"Total Length:",self.global_counter)
        self.total_reward = 0.

        self.predict_reward=0.

        self.action_c=np.zeros((self.net.num_outputs), np.int64)

      else:
        self.total_reward += reward

      # Save step
      step = Step(self.state, action, reward, next_state, value, done)
      steps.append(step)


      # Increase local and global counters
      global_step = next(self.global_counter)


      if done:
        self.state = repeat_frame(self.img_transformer.transform(self.env.reset()))
        self.d_state=repeat_frame(self.env.reset())
        break
      else:
        self.state = next_state
        self.d_state=next_s
    return steps, global_step

  def update(self, steps, sess):
    """
    Updates global policy and value networks using the local networks' gradients
    """

    # In order to accumulate the total return
    # We will use V_hat(s') to predict the future returns
    # But we will use the actual rewards if we have them
    # Ex. if we have s1, s2, s3 with rewards r1, r2, r3
    # Then G(s3) = r3 + V(s4)
    #      G(s2) = r2 + r3 + V(s4)
    #      G(s1) = r1 + r2 + r3 + V(s4)
    reward = 0.0

    if not steps[-1].done:
      _,reward=self.sample_action(steps[-1].next_state,sess)
    # Accumulate minibatch samples
    states = []
    advantages = []
    value_targets = []
    actions = []

    # loop through steps in reverse order
    for step in reversed(steps):
      reward = step.reward + self.discount_factor * reward
      advantage = reward - step.expected_value
      self.A_ad += advantage
      # Accumulate updates
      states.append(step.state)
      actions.append(step.action)
      advantages.append(advantage)
      value_targets.append(reward)



    feed_dict = {
      self.net.states: np.array(states),
      self.net.advantage: advantages,
      self.net.actions: actions,
      self.net.targets: value_targets,

    }

    # Train the global estimators using local gradients
    global_step,selected_action_probs,entropy, policy_loss,vhat,value_loss,_= sess.run([
      self.global_step,
      self.net.selected_action_probs,
      self.net.entropy,
      self.net.loss,
      self.net.vhat,
      self.net.v_loss,
      self.net_train_op,
    ], feed_dict)
    with open("Image/" + self.name + "/" + str(self.e_c) + "_loss.txt", "a") as f:
      # for word in words:
      f.write(str(actions) + " \n " + str(advantages) + " \n " + str(value_targets) +
              " \n "+str(selected_action_probs) + " \n "+str(entropy) + " \n "+str(policy_loss)
              + " \n "+str(value_loss) + " \n "+str(vhat)+"\n")
      f.write("\n")
    # Theoretically could plot these later
    return policy_loss,value_loss,advantages,actions,value_targets