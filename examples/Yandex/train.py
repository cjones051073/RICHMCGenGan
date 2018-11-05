#! /usr/bin/env python

# coding: utf-8

# In[31]:


#get_ipython().run_line_magic('env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID')
#get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=0')


# In[2]:


MODEL_NAME = "FastFastRICH_Cramer_Kaon_Test"


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
import tensorflow as tf
import keras
import keras.layers as ll
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


gpu_options = tf.GPUOptions(allow_growth=True)
tf_config = tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)


# In[5]:


import utils_rich


# In[6]:


TYPE = 'kaon'
data_train, data_val, scaler = utils_rich.get_merged_typed_dataset(TYPE, dtype=np.float32, log=True)


# In[7]:


for col in data_train.columns:
    plt.hist(data_train[col], bins=100)
    plt.title(col)
    plt.show()


# In[8]:


for col in data_val.columns:
    plt.hist(data_val[col], bins=100)
    plt.title(col)
    plt.show()


# In[9]:


BATCH_SIZE = int(5e4)
LATENT_DIMENSIONS = 64
tf.reset_default_graph()


# In[10]:


def get_dense(num_layers):
    return [ll.Dense(128, activation='relu') for i in range(num_layers)]

CRAMER_DIM = 256

critic = keras.models.Sequential(
        [ll.InputLayer([data_train.shape[1] - 1])] + get_dense(10) +
            [ll.Dense(CRAMER_DIM)])

generator = keras.models.Sequential(
        [ll.InputLayer([LATENT_DIMENSIONS + data_train.shape[1] - 1 - utils_rich.y_count])] + get_dense(10) +
            [ll.Dense(utils_rich.y_count)])


# In[11]:


plt.hist(data_train.values[:,-1], bins=100)


# In[12]:


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[13]:


train_full_ = utils_rich.get_tf_dataset(data_train, BATCH_SIZE)
train_x_1_  = utils_rich.get_tf_dataset(data_train.values[:, utils_rich.y_count:], BATCH_SIZE)
train_x_2_  = utils_rich.get_tf_dataset(data_train.values[:, utils_rich.y_count:], BATCH_SIZE)

train_full, w_full = train_full_[:,:-1], train_full_[:,-1]
train_x_1 , w_x_1  = train_x_1_ [:,:-1], train_x_1_ [:,-1]
train_x_2 , w_x_2  = train_x_2_ [:,:-1], train_x_2_ [:,-1]


# In[14]:


def cramer_critic(x, y):
    discriminated_x = critic(x)
    return tf.norm(discriminated_x - critic(y), axis=1) - tf.norm(discriminated_x, axis=1)


# In[15]:


noise_1          = tf.random_normal([tf.shape(train_x_1)[0], LATENT_DIMENSIONS], name='noise')
noise_2          = tf.random_normal([tf.shape(train_x_2)[0], LATENT_DIMENSIONS], name='noise')
generated_y_1    = generator(tf.concat([noise_1, train_x_1], axis=1))
generated_full_1 = tf.concat([generated_y_1, train_x_1], axis=1)
generated_y_2    = generator(tf.concat([noise_2, train_x_2], axis=1))
generated_full_2 = tf.concat([generated_y_2, train_x_2], axis=1)


# In[16]:


generator_loss = tf.reduce_mean(cramer_critic(train_full      , generated_full_2) * w_full * w_x_2
                              - cramer_critic(generated_full_1, generated_full_2) * w_x_1  * w_x_2)


# In[17]:


with tf.name_scope("gradient_loss"):
    alpha             = tf.random_uniform(shape=[tf.shape(train_full)[0], 1], minval=0., maxval=1.)
    interpolates      = alpha*train_full + (1.-alpha)*generated_full_1
    disc_interpolates = cramer_critic(interpolates, generated_full_2)
    gradients         = tf.gradients(disc_interpolates, [interpolates])[0]
    slopes            = tf.norm(tf.reshape(gradients, [tf.shape(gradients)[0], -1]), axis=1)
    gradient_penalty  = tf.reduce_mean(tf.square(tf.maximum(tf.abs(slopes) - 1, 0)))


# In[18]:


tf_iter         = tf.Variable(initial_value=0, dtype=tf.int32)
lambda_tf       = 20 / np.pi * 2 * tf.atan(tf.cast(tf_iter, tf.float32)/1e4)
critic_loss     = lambda_tf*gradient_penalty - generator_loss
learning_rate   = tf.train.exponential_decay(1e-3, tf_iter, 200, 0.98)
optimizer       = tf.train.RMSPropOptimizer(learning_rate)
critic_train_op = optimizer.minimize(critic_loss, var_list=critic.trainable_weights)
generator_train_op = tf.group(
    optimizer.minimize(generator_loss, var_list=generator.trainable_weights),
    tf.assign_add(tf_iter, 1))


# In[19]:


tf.summary.scalar("critic_loss", tf.reshape(critic_loss, []))
tf.summary.scalar("generator_loss", tf.reshape(generator_loss, []))
tf.summary.scalar("learning_rate", learning_rate)
tf.summary.scalar("lambda", lambda_tf)
merged_summary = tf.summary.merge_all()


# In[20]:


type(data_val)


# In[21]:


plt.hist(data_val.values[:,-1], bins=100);


# In[22]:


N_VAL = int(3e5)
validation_np = data_val.sample(N_VAL).values


# In[23]:


var_init = tf.global_variables_initializer()
weights_saver = tf.train.Saver()
tf.get_default_graph().finalize()


# In[32]:


LOGDIR = "/mnt/amaevskiy/tmp/tensorboard_logdir"
CRITIC_ITERATIONS_CONST = 15
CRITIC_ITERATIONS_VAR = 0
TOTAL_ITERATIONS = int(1e5)
VALIDATION_INTERVAL = 100
MODEL_WEIGHTS_FILE = "weights/%s.ckpt" % MODEL_NAME
train_writer = tf.summary.FileWriter(os.path.join(LOGDIR, MODEL_NAME, "train"))
test_writer = tf.summary.FileWriter(os.path.join(LOGDIR, MODEL_NAME, "test"))
critic_policy = lambda i: (
    CRITIC_ITERATIONS_CONST + (CRITIC_ITERATIONS_VAR * (TOTAL_ITERATIONS - i)) // TOTAL_ITERATIONS)
with tf.Session(config=tf_config) as sess:
    sess.run(var_init)
    try:
        weights_saver.restore(sess, MODEL_WEIGHTS_FILE)
    except tf.errors.NotFoundError:
        print("Can't restore parameters: no file with weights")
    for i in range(TOTAL_ITERATIONS):
        for j in range(critic_policy(i)):
            sess.run(critic_train_op)
        train_summary, _, interation = sess.run([merged_summary, generator_train_op, tf_iter])
        train_writer.add_summary(train_summary, interation)
        if i % VALIDATION_INTERVAL == 0:
            clear_output(False)
            test_summary, test_generated = sess.run([merged_summary, generated_y_1], {
                train_x_1_: validation_np[:, utils_rich.y_count:],
                train_x_2_: validation_np[:, utils_rich.y_count:], train_full_: validation_np})
            test_writer.add_summary(test_summary, interation)
            weights_saver.save(sess, MODEL_WEIGHTS_FILE)
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            for INDEX, ax in zip((0, 1, 3, 4), axes.flatten()):
                _, bins, _ = ax.hist(validation_np[:, INDEX], bins=100, label="data", normed=True,
                                     weights=validation_np[:,-1])
                ax.hist(test_generated[:, INDEX], bins=bins, label="generated", alpha=0.5, normed=True,
                        weights=validation_np[:,-1])
                ax.legend()
                ax.set_title(utils_rich.dll_columns[INDEX])
            plt.show()


# In[35]:


with tf.Session(config=tf_config) as sess:
    sess.run(var_init)
    weights_saver.restore(sess, MODEL_WEIGHTS_FILE)
    sess.graph._unsafe_unfinalize()
    tf.saved_model.simple_save(sess, os.path.join(
        "exported_model", MODEL_NAME), inputs={"x": train_x_1_}, outputs={"dlls": generated_y_1})
    tf.get_default_graph().finalize()
from sklearn.externals import joblib
joblib.dump(scaler, os.path.join('preprocessors', MODEL_NAME) + "_preprocessor.pkl")


# In[23]:


#joblib.dump(scaler, os.path.join('preprocessors', MODEL_NAME) + "_preprocessor.pkl")


# Timing!
