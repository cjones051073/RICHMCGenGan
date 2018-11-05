#! /usr/bin/env python

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
import tensorflow as tf
import keras
import keras.layers as ll
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
import pandas as pd
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os, shutil
from IPython.display import clear_output
import scipy
import platform
from tqdm import tqdm
   
import RICH

# To make sure that we can reproduce the experiment and get the same results
np.random.seed(1234)

tf_config = tf.ConfigProto()
#tf_config.gpu_options = tf.GPUOptions(allow_growth=True)
#tf_config.log_device_placement=True
#tf_config.intra_op_parallelism_threads = 16
#tf_config.inter_op_parallelism_threads = 16
tf.reset_default_graph()

# Job size parameters

MODEL_NAME = "Cramer_Test2"

# Standard large job
#maxData                 = -1
#BATCH_SIZE              = int(5e4)
#VALIDATION_SIZE         = int(3e5)
#TOTAL_ITERATIONS        = int(1e4)
#VALIDATION_INTERVAL     = 100

# medium job
#maxData                 = -1
#BATCH_SIZE              = int(1e3)
#VALIDATION_SIZE         = int(1e2)
#TOTAL_ITERATIONS        = int(1e3)
#VALIDATION_INTERVAL     = 100

# tiny test job
maxData                 = int(1e3)
BATCH_SIZE              = int(1e2)
VALIDATION_SIZE         = int(1e2)
TOTAL_ITERATIONS        = int(1e2)
VALIDATION_INTERVAL     = 10

print ( "BATCH SIZE         ", BATCH_SIZE )
print ( "VALIDATION SIZE    ", VALIDATION_SIZE )
print ( "# ITERATONS        ", TOTAL_ITERATIONS )
print ( "VALIDATION INTERVAL", VALIDATION_INTERVAL )

CRAMER_DIM         = 256
N_LAYERS_CRITIC    = 10
N_LAYERS_GENERATOR = 10

print ( "CRITIC LAYERS      ", N_LAYERS_CRITIC )
print ( "GENERATOR LAYERS   ", N_LAYERS_GENERATOR )
print ( "CRAMER OUTPUT DIM  ", CRAMER_DIM )

#LEAK_RATE    = 0.2
#DROPOUT_RATE = 0.3
LEAK_RATE    = 0
DROPOUT_RATE = 0

print ( "LEAK RATE          ", LEAK_RATE )
print ( "DROPOUT RATE       ", DROPOUT_RATE )

# functor to give the number of training runs per iteration
CRITIC_ITERATIONS_CONST = 15
CRITIC_ITERATIONS_VAR   = 0
critic_policy = lambda i: (
    CRITIC_ITERATIONS_CONST + (CRITIC_ITERATIONS_VAR * (TOTAL_ITERATIONS - i)) // TOTAL_ITERATIONS)

plots_dir = "plots/"+platform.node()+"/Tf/"+MODEL_NAME+"/"+str(TOTAL_ITERATIONS)+"-its/"
if     os.path.exists(plots_dir) : shutil.rmtree(plots_dir) 
if not os.path.exists(plots_dir) : os.makedirs(plots_dir)
LOGDIR = plots_dir

# Total input dimensions of generator (including noise)
GENERATOR_DIMENSIONS = 64

# names of variables to extract for training data
train_names = [ 'NumRich1Hits', 'NumRich2Hits', 'TrackP', 'TrackPt' 
                #,'NumPVs'
                #,"NumLongTracks"
                #,"TrackChi2PerDof", "TrackNumDof"
                #,'TrackVertexX', 'TrackVertexY', 'TrackVertexZ' 
                #,'TrackRich1EntryX', 'TrackRich1EntryY' 
                #,'TrackRich1ExitX', 'TrackRich1ExitY'
                #,'TrackRich2EntryX', 'TrackRich2EntryY'
                #,'TrackRich2ExitX', 'TrackRich2ExitY' 
]
# Train on random input only
#train_names = [ ]

# names for target data
target_names = [ 'RichDLLe', 'RichDLLmu', 'RichDLLk', 'RichDLLp', 'RichDLLd', 'RichDLLbt' ]
output_dim = len(target_names)

# amount of noise to add to training data
NOISE_DIMENSIONS = GENERATOR_DIMENSIONS - len(train_names)

# Split data into train and validation samples
data_raw, val_raw = train_test_split( RICH.createLHCbData( train_names+target_names,  maxData, 'KAONS' ),
                                      random_state = 1234 )

# scale
print ( "Scaling the data" )
ndataforscale = min( 500000, data_raw.shape[0] )
scaler     = RICH.getScaler( data_raw.iloc[0:ndataforscale] )
dll_scaler = RICH.getScaler( data_raw[target_names].iloc[0:ndataforscale] )
data_train = pd.DataFrame( scaler.transform(data_raw), columns = data_raw.columns, dtype=np.float32 )
data_val   = pd.DataFrame( scaler.transform(val_raw),  columns = val_raw.columns,  dtype=np.float32 )
print( "Training Norm Data\n", data_train.head() )

# Make some input / output plots
RICH.plots( "output_raw",  data_raw[target_names],   plots_dir )
RICH.plots( "inputs_raw",  data_raw[train_names],    plots_dir )
RICH.plots( "output_norm", data_train[target_names], plots_dir )
RICH.plots( "inputs_norm", data_train[train_names],  plots_dir )

n_input_layer = data_train.shape[1]
print( "Building Critic, #inputs=", n_input_layer )
critic = keras.models.Sequential()
critic.add( ll.InputLayer( [ n_input_layer ] ) )
for i in range(0,N_LAYERS_CRITIC) :
    critic.add( ll.Dense(128, activation='relu') )
    if LEAK_RATE    > 0 : critic.add( ll.LeakyReLU(LEAK_RATE) )
    if DROPOUT_RATE > 0 : critic.add( ll.Dropout(DROPOUT_RATE) )
critic.add( ll.Dense(CRAMER_DIM) )
critic.summary()

print( "Building Generator, #inputs=", GENERATOR_DIMENSIONS )
generator = keras.models.Sequential()
generator.add( ll.InputLayer( [GENERATOR_DIMENSIONS] ) )
for i in range(0,N_LAYERS_GENERATOR) :
    generator.add( ll.Dense(128, activation='relu') )
    if LEAK_RATE    > 0 : generator.add( ll.LeakyReLU(LEAK_RATE) )
    if DROPOUT_RATE > 0 : generator.add( ll.Dropout(DROPOUT_RATE) )
generator.add( ll.Dense(output_dim) )
generator.summary()

# Create tensor data iterators

# everything, inputs and outputs
train_full = RICH.get_tf_dataset(data_train, BATCH_SIZE)
# Inputs only, two sets
train_x_1  = RICH.get_tf_dataset(data_train[train_names].values, BATCH_SIZE)
train_x_2  = RICH.get_tf_dataset(data_train[train_names].values, BATCH_SIZE)

# Create noise data
noise_1          = tf.random_normal([tf.shape(train_x_1)[0], NOISE_DIMENSIONS], name='noise')
noise_2          = tf.random_normal([tf.shape(train_x_2)[0], NOISE_DIMENSIONS], name='noise')
generated_y_1    = generator( tf.concat([noise_1, train_x_1], axis=1) )
generated_full_1 = tf.concat([generated_y_1, train_x_1], axis=1)
generated_y_2    = generator(tf.concat([noise_2, train_x_2], axis=1))
generated_full_2 = tf.concat([generated_y_2, train_x_2], axis=1)

def cramer_critic(x, y):
    discriminated_x = critic(x)
    return tf.norm(discriminated_x - critic(y), axis=1) - tf.norm(discriminated_x, axis=1)

generator_loss = tf.reduce_mean(cramer_critic(train_full      , generated_full_2) 
                              - cramer_critic(generated_full_1, generated_full_2) )

with tf.name_scope("gradient_loss"):
    alpha             = tf.random_uniform(shape=[tf.shape(train_full)[0], 1], minval=0., maxval=1.)
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

tf.summary.scalar("critic_loss",    tf.reshape(critic_loss, []))
tf.summary.scalar("generator_loss", tf.reshape(generator_loss, []))
tf.summary.scalar("learning_rate",  learning_rate)
tf.summary.scalar("lambda",         lambda_tf)
merged_summary = tf.summary.merge_all()

validation_np     = data_val.sample(VALIDATION_SIZE)
validation_np_raw = val_raw.sample(VALIDATION_SIZE)

var_init      = tf.global_variables_initializer()
weights_saver = tf.train.Saver()
tf.get_default_graph().finalize()

MODEL_WEIGHTS_FILE      = plots_dir+"weights/%s.ckpt" % MODEL_NAME
train_writer            = tf.summary.FileWriter(os.path.join(LOGDIR, "summary", "train"))
test_writer             = tf.summary.FileWriter(os.path.join(LOGDIR, "summary", "test"))

with tf.Session(config=tf_config) as sess:

    # Initialise
    sess.run(var_init)
    
    # Try and restore a saved weights file
    try:
        weights_saver.restore(sess, MODEL_WEIGHTS_FILE)
    except tf.errors.NotFoundError:
        print("Can't restore parameters: no file with weights")

    # Do the iterations
    for i in tqdm(range(TOTAL_ITERATIONS)):

        for j in range(critic_policy(i)) :
            sess.run(critic_train_op)

        train_summary, _, interation = sess.run([merged_summary, generator_train_op, tf_iter])
        train_writer.add_summary(train_summary, interation)

        # Do validation now and then
        if i % VALIDATION_INTERVAL == 0:

            # Directory for plots etc. for this iteratons
            it_dir = plots_dir+"iteration/"+str( '%06d' % i )+"/"
            if not os.path.exists(it_dir) : os.makedirs(it_dir)

            clear_output(False)
            test_summary, test_generated = sess.run( [merged_summary, generated_y_1], {
                train_x_1 : validation_np[train_names].values,
                train_x_2 : validation_np[train_names].values,
                train_full: validation_np.values } )

            # Summary and weights
            test_writer.add_summary(test_summary, interation)
            weights_saver.save(sess, MODEL_WEIGHTS_FILE)

            # normalised generated DLLs
            ix,iy    = RICH.plotDimensions(output_dim)
            fig,axes = plt.subplots(ix, iy, figsize=(15, 15))
            for INDEX, ax in zip( range(0,output_dim), axes.flatten() ):
                _, bins, _ = ax.hist(validation_np[target_names].values[:, INDEX],
                                     bins=100, label="data", density=True)
                ax.hist(test_generated[:, INDEX], bins=bins, label="generated", alpha=0.5, density=True)
                ax.legend()
                ax.set_title(target_names[INDEX])
            plt.savefig(it_dir+"dlls-norm.png")
            plt.close()
            
            # raw generated DLLs
            fig,axes = plt.subplots(ix, iy, figsize=(15, 15))
            test_generated_raw = dll_scaler.inverse_transform( test_generated )
            for INDEX, ax in zip( range(0,output_dim), axes.flatten() ):
                _, bins, _ = ax.hist(validation_np_raw[target_names].values[:, INDEX],
                                     bins=100, label="data", density=True)
                ax.hist(test_generated_raw[:, INDEX], bins=bins, label="generated", alpha=0.5, density=True)
                ax.legend()
                ax.set_title(target_names[INDEX])
            plt.savefig(it_dir+"dlls-raw.png")
            plt.close()

            # DLL correlations
            RICH.outputCorrs( "correlations", test_generated, validation_np[target_names].values,
                              target_names, it_dir )


with tf.Session(config=tf_config) as sess:
    sess.run(var_init)
    weights_saver.restore(sess, MODEL_WEIGHTS_FILE)
    sess.graph._unsafe_unfinalize()
    tf.saved_model.simple_save(sess,
                               os.path.join(plots_dir,"exported_model"),
                               inputs={"x": train_x_1}, outputs={"dlls": generated_y_1})
    tf.get_default_graph().finalize()
    
#from sklearn.externals import joblib
#joblib.dump(scaler, os.path.join(plots_dir, 'preprocessors', MODEL_NAME) + "_preprocessor.pkl")
