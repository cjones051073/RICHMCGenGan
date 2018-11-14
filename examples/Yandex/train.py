#! /usr/bin/env python3

import os, shutil, sys
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
import keras
import keras.layers as ll
import pandas as pd
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output
import scipy
import argparse

parser = argparse.ArgumentParser(description='Job Parameters')

parser.add_argument( '--debug', action='store_true' )
parser.set_defaults(debug=False)

args,unparsed = parser.parse_known_args()
print( "Job arguments", args )

if args.debug :
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf

tf.reset_default_graph()

if args.debug :
    sess = tf.InteractiveSession()

# To make sure that we can reproduce the experiment and get the same results
RNDM_SEED = 12345
np.random.seed(RNDM_SEED)
tf.set_random_seed(RNDM_SEED)

MODEL_NAME = "Yandex-Test1"

LOGDIR = "/home/jonesc/Projects/RICHMCGenGan/output/"+MODEL_NAME+"/"
if     os.path.exists(LOGDIR) : shutil.rmtree(LOGDIR) 
if not os.path.exists(LOGDIR) : os.makedirs(LOGDIR)

gpu_options = tf.GPUOptions( allow_growth=True )
tf_config   = tf.ConfigProto(log_device_placement=False,
                             gpu_options=gpu_options,
                             allow_soft_placement = True)

import utils_rich

TYPE = 'kaon'
data_train, data_val, scaler = utils_rich.get_merged_typed_dataset(TYPE, dtype=np.float32, log=True)

if args.debug : 
    print( "norm train data\n", data_train.tail() )
    print( "norm data val\n", data_val.tail() )

for col in data_train.columns:
    plt.hist(data_train[col], bins=100)
    plt.title(col)
    #plt.show()
    plt.savefig(LOGDIR+col+'.png')


for col in data_val.columns:
    plt.hist(data_val[col], bins=100)
    plt.title(col)
    #plt.show()
    plt.savefig(LOGDIR+col+'.png')

BATCH_SIZE = int(5e4)
LATENT_DIMENSIONS = 64

def get_dense(num_layers):
    return [ll.Dense(128, activation='relu') for i in range(num_layers)]

CRAMER_DIM = 256

tf_device = '/gpu:2'
if args.debug : tf_device = '/cpu:0'
with tf.device(tf_device):

    n_input_layer = data_train.shape[1] - 1 # subtract one as weights not used
    print( "Building Critic, #inputs=", n_input_layer )
    critic = keras.models.Sequential(
        [ll.InputLayer([n_input_layer])] + get_dense(10) +
        [ll.Dense(CRAMER_DIM)])
    critic.summary()
    
    GENERATOR_DIMENSIONS = LATENT_DIMENSIONS + (data_train.shape[1]-1) - utils_rich.y_count
    print( "Building Generator, #inputs=", GENERATOR_DIMENSIONS )
    generator = keras.models.Sequential(
        [ll.InputLayer([GENERATOR_DIMENSIONS])] + get_dense(10) +
        [ll.Dense(utils_rich.y_count)])
    generator.summary()
    
    plt.hist(data_train.values[:,-1], bins=100)
    
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot
    
    train_full_ = utils_rich.get_tf_dataset(data_train, BATCH_SIZE, seed = 15346)
    train_full, w_full = train_full_[:,:-1], train_full_[:,-1]

    train_x_1_all = utils_rich.get_tf_dataset(data_train.values, BATCH_SIZE, seed = 1111)
    #train_x_1_  = utils_rich.get_tf_dataset(data_train.values[:, utils_rich.y_count:], BATCH_SIZE, seed = 1111)
    train_x_1_ =  train_x_1_all[:, utils_rich.y_count:]
    train_x_1 , w_x_1  = train_x_1_ [:,:-1], train_x_1_ [:,-1]
    train_x_1_targets  = train_x_1_all[:, :utils_rich.y_count]
      
    train_x_2_  = utils_rich.get_tf_dataset(data_train.values[:, utils_rich.y_count:], BATCH_SIZE, seed = 2222)
    train_x_2 , w_x_2  = train_x_2_ [:,:-1], train_x_2_ [:,-1]
    
    if args.debug :
        print( "train_full\n", train_full.shape, "\n", train_full.eval() )
        print( "train_x_1\n", train_x_1.shape, "\n", train_x_1.eval() )
        print( "train_x_2\n", train_x_2.shape, "\n", train_x_2.eval() )
        print( "w_x_1\n", w_x_1.shape, "\n", w_x_1.eval() )
        print( "w_x_2\n", w_x_2.shape, "\n", w_x_2.eval() )
        print( "train_x_1_targets\n", train_x_1_targets.shape, "\n", train_x_1_targets.eval() )
        
    def cramer_critic(x, y):
        discriminated_x = critic(x)
        return tf.norm(discriminated_x - critic(y), axis=1) - tf.norm(discriminated_x, axis=1)
    
    print( "Creating noise data" )
    noise_1          = tf.random_normal([tf.shape(train_x_1)[0], LATENT_DIMENSIONS], name='noise', seed = 4242 )
    noise_2          = tf.random_normal([tf.shape(train_x_2)[0], LATENT_DIMENSIONS], name='noise', seed = 2424 )
    
    if args.debug :
        print("noise_1\n", noise_1.shape, '\n', noise_1.eval() )
        print("noise_2\n", noise_2.shape, '\n', noise_2.eval() )
        
    generated_y_1    = generator(tf.concat([noise_1, train_x_1], axis=1))
    generated_y_2    = generator(tf.concat([noise_2, train_x_2], axis=1))
    
    generated_full_1 = tf.concat([generated_y_1, train_x_1], axis=1)
    generated_full_2 = tf.concat([generated_y_2, train_x_2], axis=1)
    
    if args.debug :
        print( "generated_y_1",    generated_y_1.shape )
        print( "generated_y_2",    generated_y_2.shape )
        print( "generated_full_1", generated_full_1.shape )
        print( "generated_full_2", generated_full_2.shape )
        
    generator_loss = tf.reduce_mean(cramer_critic(train_full      , generated_full_2) * w_full * w_x_2
                                  - cramer_critic(generated_full_1, generated_full_2) * w_x_1  * w_x_2)
    
    with tf.name_scope("gradient_loss"):
        alpha             = tf.random_uniform(shape=[tf.shape(train_full)[0], 1], minval=0., maxval=1., seed = 5678 )
        interpolates      = alpha*train_full + (1.-alpha)*generated_full_1
        disc_interpolates = cramer_critic(interpolates, generated_full_2)
        gradients         = tf.gradients(disc_interpolates, [interpolates])[0]
        slopes            = tf.norm(tf.reshape(gradients, [tf.shape(gradients)[0], -1]), axis=1)
        gradient_penalty  = tf.reduce_mean(tf.square(tf.maximum(tf.abs(slopes) - 1, 0)))
    
    tf_iter         = tf.Variable(initial_value=0, dtype=tf.int32)
    lambda_tf       = 20 / np.pi * 2 * tf.atan(tf.cast(tf_iter, tf.float32)/1e4)
    critic_loss     = lambda_tf*gradient_penalty - generator_loss
    learning_rate   = tf.train.exponential_decay(1e-3, tf_iter, 200, 0.98)
    
    optimizer       = tf.train.RMSPropOptimizer(learning_rate)
    critic_train_op = optimizer.minimize(critic_loss, var_list=critic.trainable_weights)
    
    generator_train_op = tf.group(
        optimizer.minimize(generator_loss, var_list=generator.trainable_weights),
        tf.assign_add(tf_iter, 1))

    if args.debug :
        print("alpha\n", alpha.shape, '\n', alpha.eval() )

N_VAL = int(3e5)
validation_np = data_val.sample(N_VAL)

if args.debug :
    print( "validation_np\n", validation_np.shape, "\n", validation_np.tail() )
    
target_diff = generated_y_1 - train_x_1_targets
if args.debug :
    print( "target_diff\n", target_diff.shape )
    
tf.summary.scalar("critic_loss", tf.reshape(critic_loss, []))
tf.summary.scalar("generator_loss", tf.reshape(generator_loss, []))
tf.summary.scalar("learning_rate", learning_rate)
tf.summary.scalar("lambda", lambda_tf)

# Add histograms for generated output
tf.summary.histogram( "interpolates", interpolates )
tf.summary.histogram( "gradients", gradients )
tf.summary.histogram( "slopes", slopes )

for iname in range(utils_rich.y_count) :
    tf.summary.histogram( utils_rich.dll_columns[iname], generated_y_1[:,iname] )
    tf.summary.histogram( utils_rich.dll_columns[iname]+"-Diff", target_diff[:,iname] )

merged_summary = tf.summary.merge_all()

# if args.debug, abort before starting training..
if args.debug : sys.exit(0)

#type(data_val)
#plt.hist(data_val.values[:,-1], bins=100);

var_init = tf.global_variables_initializer()
weights_saver = tf.train.Saver()
tf.get_default_graph().finalize()

CRITIC_ITERATIONS_CONST = 15
CRITIC_ITERATIONS_VAR = 0
TOTAL_ITERATIONS = int(1e5)
#TOTAL_ITERATIONS = int(1e2)
VALIDATION_INTERVAL = 100
#VALIDATION_INTERVAL  = 10
TRAIN_WRITE_INTERVAL = 25
#TRAIN_WRITE_INTERVAL = 5
MODEL_WEIGHTS_FILE = LOGDIR+"weights/%s.ckpt" % MODEL_NAME
train_writer = tf.summary.FileWriter(os.path.join(LOGDIR, "summary", "train"))
test_writer  = tf.summary.FileWriter(os.path.join(LOGDIR, "summary", "test"))

critic_policy = lambda i: (
    CRITIC_ITERATIONS_CONST + (CRITIC_ITERATIONS_VAR * (TOTAL_ITERATIONS - i)) // TOTAL_ITERATIONS)

# To make sure that we can reproduce the experiment and get the same results
np.random.seed(RNDM_SEED)
tf.set_random_seed(RNDM_SEED)

with tf.Session(config=tf_config) as sess:
    
    sess.run(var_init)
    
    try:
        weights_saver.restore(sess, MODEL_WEIGHTS_FILE)
    except tf.errors.NotFoundError:
        print("Can't restore parameters: no file with weights")
        
    for i in tqdm(range(1,1+TOTAL_ITERATIONS)):
        
        for j in range(critic_policy(i)):
            sess.run(critic_train_op)

        train_summary, _, interation = sess.run([merged_summary, generator_train_op, tf_iter])

        if i % TRAIN_WRITE_INTERVAL == 0 or i == 1 :
            train_writer.add_summary(train_summary, interation)

        if i % VALIDATION_INTERVAL == 0 or i == 1 :
            clear_output(False)
            
            test_summary, test_generated = sess.run([merged_summary, generated_y_1], {
                train_x_1_    : validation_np.values[:, utils_rich.y_count:],
                train_x_2_    : validation_np.values[:, utils_rich.y_count:],
                train_x_1_all : validation_np.values,
                train_full_   : validation_np.values})

            test_writer.add_summary(test_summary, interation)
            weights_saver.save(sess, MODEL_WEIGHTS_FILE)
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            for INDEX, ax in zip((0, 1, 3, 4), axes.flatten()):
                _, bins, _ = ax.hist(validation_np.values[:, INDEX], bins=100, label="data", density=True)
                                     #weights=validation_np.values[:,-1])
                ax.hist(test_generated[:, INDEX], bins=bins, histtype='stepfilled', label="generated",
                        alpha=0.5, density=True)
                        #weights=validation_np.values[:,-1])
                ax.legend()
                ax.set_title(utils_rich.dll_columns[INDEX])
            #plt.show()
            plt.savefig(LOGDIR+'generated-it'+str(i)+'.png')


with tf.Session(config=tf_config) as sess:
    sess.run(var_init)
    weights_saver.restore(sess, MODEL_WEIGHTS_FILE)
    sess.graph._unsafe_unfinalize()
    tf.saved_model.simple_save(sess, os.path.join(
        "exported_model", MODEL_NAME), inputs={"x": train_x_1_}, outputs={"dlls": generated_y_1})
    tf.get_default_graph().finalize()
    
#from sklearn.externals import joblib
#joblib.dump(scaler, os.path.join((LOGDIR, 'preprocessors', MODEL_NAME) + "_preprocessor.pkl")

