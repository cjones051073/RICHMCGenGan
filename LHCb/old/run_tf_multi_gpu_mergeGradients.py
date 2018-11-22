#! /usr/bin/env python3

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
#import matplotlib.pyplot as plt
import os, sys, platform
from tqdm import tqdm
import RICH
import argparse
import keras
from sklearn.model_selection import train_test_split
import pandas as pd

print( "Running on", platform.node() )

parser = argparse.ArgumentParser(description='Job Parameters')

parser.add_argument( '--name', type=str, default="PPtR1R2HitsR12EntryExit-MergeGrads" )

parser.add_argument( '--outputdir', type=str, default="/home/jonesc/Projects/RICHMCGenGan/output" )

parser.add_argument( '--datareadsize', type=int, default="4000000" )

parser.add_argument( '--niterations', type=int, default="100000" )

parser.add_argument( '--validationsize', type=int, default="300000" )
parser.add_argument( '--valfrac', type=float, default="0.2" )

parser.add_argument( '--batchsize', type=int, default="50000" )

parser.add_argument( '--validationinterval', type=int, default="100" )
parser.add_argument( '--trainwriteinterval', type=int, default="50" )

parser.add_argument( '--ngpus', type=int, default="3" )

parser.add_argument( '--datadir', type=str, default="/home/jonesc/Projects/RICHMCGenGan/data" )

parser.add_argument( '--ncriticlayers', type=int, default="10" )
parser.add_argument( '--ngeneratorlayers', type=int, default="10" )

parser.add_argument( '--leakrate', type=float, default="0.0" )
parser.add_argument( '--dropoutrate', type=float, default="0.0" )
parser.add_argument( '--cramerdim', type=int, default="256" )

parser.add_argument( '--inputvars', type=str, nargs='+',
                     default = [  'TrackP', 'TrackPt'
                                 ,'NumRich1Hits', 'NumRich2Hits'
                                 ,'TrackRich1EntryX', 'TrackRich1EntryY'
                                 ,'TrackRich1ExitX', 'TrackRich1ExitY'
                                 ,'TrackRich2EntryX', 'TrackRich2EntryY'
                                 ,'TrackRich2ExitX', 'TrackRich2ExitY' ] )

parser.add_argument( '--outputvars', type=str, nargs='+',
                     default = ['RichDLLe', 'RichDLLk', 'RichDLLmu',
                                'RichDLLp', 'RichDLLd', 'RichDLLbt'] )

parser.add_argument( '--batchmode', action='store_true' )
parser.set_defaults(batchmode=False)

parser.add_argument( '--debug', action='store_true' )
parser.set_defaults(debug=False)

args,unparsed = parser.parse_known_args()
print( "Job arguments", args )

MODEL_NAME          = args.name

if args.dropoutrate > 0 : MODEL_NAME += "-Drop"+str(args.dropoutrate)
if args.leakrate    > 0 : MODEL_NAME += "-Leak"+str(args.leakrate)

TOTAL_ITERATIONS    = args.niterations

if args.debug :
  os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf

tf.reset_default_graph()
# debugging
if args.debug : sess = tf.InteractiveSession()

# To make sure that we can reproduce the experiment and get the same results
RNDM_SEED = 12345
np.random.seed(RNDM_SEED)
tf.set_random_seed(RNDM_SEED)

if not args.debug and platform.node() == 'gorfrog' :
  print("Using GPU options")
  tf_config = tf.ConfigProto( gpu_options = tf.GPUOptions(allow_growth=False),
                              allow_soft_placement = True )
else:
  tf_config = tf.ConfigProto()
tf_config.log_device_placement         = True
tf_config.intra_op_parallelism_threads = 16
tf_config.inter_op_parallelism_threads = 16

# Job size parameters

BATCH_SIZE         = args.batchsize
maxData            = args.datareadsize

CRAMER_DIM         = args.cramerdim
N_LAYERS_CRITIC    = args.ncriticlayers
N_LAYERS_GENERATOR = args.ngeneratorlayers

LEAK_RATE          = args.leakrate
DROPOUT_RATE       = args.dropoutrate

# inputs
train_names = args.inputvars 

# names for target data
target_names = args.outputvars 

# number outputs from generator
output_dim = len(target_names)

# amount of noise to add to training data
NOISE_DIMENSIONS = 64

# Total input dimensions of generator (including noise)
GENERATOR_DIMENSIONS = NOISE_DIMENSIONS + len(train_names)

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    vars  = []
    if args.debug : print( "Found", len(grad_and_vars), "grads" )
    for g, v in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      if g is not None :
        if args.debug :
          print( "arg,var", g, v )
        expanded_g = tf.expand_dims(g, 0)
        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)
        # append to vars
        vars.append(v)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = vars[0]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)

  # return the final average grads
  return average_grads

with tf.device('/cpu:0'):

  # Create a variable to count the number of train() calls. 
  g_step = tf.train.get_or_create_global_step()

  # read data from file
  readData = RICH.createLHCbData( target_names+train_names,  
                                  maxData, 'KAONS',
                                  args.datadir )

  # create data scaler
  ndataforscale = min( 1000000, readData.shape[0] )
  scaler     = RICH.getScaler( readData.iloc[0:ndataforscale] )
  #scaler     = RICH.getScaler( readData )
  dll_scaler = RICH.getScaler( readData[target_names].iloc[0:ndataforscale] )

  # split off the validation sample
  readData, val_raw = train_test_split( readData, test_size=args.valfrac, random_state=42 )
  if args.debug : 
    print( "raw train data\n", readData.tail() )

  print ( "Scaling the validation data" )
  data_val = pd.DataFrame( scaler.transform(val_raw), columns = val_raw.columns,  dtype=np.float32 )
                                         
  # split into an independent sample for each GPU
  data_raw = RICH.splitDataFrame( readData, args.ngpus )
  del readData
 
  # create the normalised training data
  print ( "Scaling the training data" )
  data_train = [ ]
  for d in data_raw :
    d_norm = pd.DataFrame( scaler.transform(d), columns = d.columns, dtype=np.float32 )
    if args.debug : 
      print( "norm train data\n", d_norm.shape, "\n", d_norm.tail() )
    data_train.append(d_norm)
    
  # Add missing 'weights' columns. Not used, but might be useful in the future.
  weight_col = 'Weight'
  data_val[weight_col] = np.float32(1.0)
  for d in data_train : d[weight_col] = np.float32(1.0)

  # Decay the learning rate exponentially based on the number of steps.
  learning_rate = tf.train.exponential_decay( 2e-3, g_step, 200, 0.985 )

  # Optimizer
  optimizer = tf.train.RMSPropOptimizer(learning_rate)

  # save the gradients
  gpu_grads_critics    = [ ]
  gpu_grads_generators = [ ]

  with tf.variable_scope(tf.get_variable_scope()):
    for igpu in range(args.ngpus):
      tfdevice = '/gpu:%d' % igpu
      if args.debug : tfdevice = '/cpu:0'
      with tf.device(tfdevice):
        with tf.name_scope( '%s_%d' % ("GPU",igpu) ) as scope:

          print( "Configuring", tfdevice, "data sample size", data_train[igpu].shape[0] )

          # Build the critic and generator models

          n_input_layer = data_train[igpu].shape[1] - 1 # subtract one as weights not used
          print( "Building Critic, #inputs=", n_input_layer )
          critic_i = keras.models.Sequential()
          critic_i.add( keras.layers.InputLayer( [ n_input_layer ] ) )
          for i in range(0,N_LAYERS_CRITIC) :
            critic_i.add( keras.layers.Dense(128, activation='relu' ) )
            if LEAK_RATE    > 0 : critic_i.add( keras.layers.LeakyReLU(LEAK_RATE) )
            if DROPOUT_RATE > 0 : critic_i.add( keras.layers.Dropout(DROPOUT_RATE) )
          critic_i.add( keras.layers.Dense(CRAMER_DIM) )
          critic_i.summary()
    
          print( "Building Generator, #inputs=", GENERATOR_DIMENSIONS )
          generator_i = keras.models.Sequential()
          generator_i.add( keras.layers.InputLayer( [GENERATOR_DIMENSIONS] ) )
          for i in range(0,N_LAYERS_GENERATOR) :
            generator_i.add( keras.layers.Dense(128, activation='relu' ) )
            if LEAK_RATE    > 0 : generator_i.add( keras.layers.LeakyReLU(LEAK_RATE) )
            if DROPOUT_RATE > 0 : generator_i.add( keras.layers.Dropout(DROPOUT_RATE) )
          generator_i.add( keras.layers.Dense(output_dim) )
          generator_i.summary()
     
          # Create tensor data iterators
          print( "Creating data iterators" )
          # everything, inputs and outputs
          train_full_ = RICH.get_tf_dataset( data_train[igpu], BATCH_SIZE, seed = 15346+igpu )
          train_full, w_full = train_full_[:,:-1], train_full_[:,-1]
  
          # Inputs only, two independent random sets
          train_x_1_all = RICH.get_tf_dataset(data_train[igpu].values, BATCH_SIZE, seed = 1111+igpu )
          #train_x_1_ = RICH.get_tf_dataset(data_train[igpu][train_names+[weight_col]].values, BATCH_SIZE, seed = 1111+igpu)
          #train_x_2_ = RICH.get_tf_dataset(data_train[igpu][train_names+[weight_col]].values, BATCH_SIZE, seed = 2222+igpu)
          #train_x_1_ = RICH.get_tf_dataset(data_train[igpu].values[:, output_dim:], BATCH_SIZE, seed = 1111+igpu)
          train_x_1_ =  train_x_1_all[:,output_dim:]
          train_x_2_ = RICH.get_tf_dataset( data_train[igpu].values[:, output_dim:], BATCH_SIZE, seed = 2222+igpu)
          #train_x_1, w_x_1 = train_x_1_[:,:-1], train_x_1_[:,-1]
          #train_x_2, w_x_2 = train_x_2_[:,:-1], train_x_2_[:,-1]
          train_x_1 = train_x_1_[:,:-1]
          train_x_2 = train_x_2_[:,:-1]
          train_x_1_targets  = train_x_1_all[:, :output_dim]
          
          if args.debug : 
            print( "train_full\n", train_full.shape, "\n", train_full.eval() )
            print( "train_x_1\n", train_x_1.shape, "\n", train_x_1.eval() )
            print( "train_x_2\n", train_x_2.shape, "\n", train_x_2.eval() )
            #print( "w_x_1\n", w_x_1.shape, "\n", w_x_1.eval() )
            #print( "w_x_2\n", w_x_2.shape, "\n", w_x_2.eval() )
            print( "train_x_1_targets\n", train_x_1_targets.shape, "\n", train_x_1_targets.eval() )
    
          # Create noise data
          print( "Creating noise data" )
          noise_1 = tf.random_normal([tf.shape(train_x_1)[0], NOISE_DIMENSIONS], name='noise', seed = 4242+igpu )
          noise_2 = tf.random_normal([tf.shape(train_x_2)[0], NOISE_DIMENSIONS], name='noise', seed = 2424+igpu )
  
          if args.debug :
            print("noise_1\n", noise_1.shape, '\n', noise_1.eval() )
            print("noise_2\n", noise_2.shape, '\n', noise_2.eval() )
    
          generated_y_1    = generator_i( tf.concat([noise_1, train_x_1], axis=1) )
          generated_y_2    = generator_i( tf.concat([noise_2, train_x_2], axis=1) )
  
          generated_full_1 = tf.concat([generated_y_1, train_x_1], axis=1)
          generated_full_2 = tf.concat([generated_y_2, train_x_2], axis=1)
  
          if args.debug :
            print( "generated_y_1",   generated_y_1.shape )
            print( "generated_y_2",   generated_y_2.shape )
            print( "generated_full_1", generated_full_1.shape )
            print( "generated_full_2", generated_full_2.shape )
        
          def cramer_critic( x, y ):
            discriminated_x = critic_i(x)
            return tf.norm(discriminated_x - critic_i(y), axis=1) - tf.norm(discriminated_x, axis=1)
  
          # loss function for generator network (when weights are all 1)
          generator_loss = tf.reduce_mean(cramer_critic(train_full      , generated_full_2) 
                                        - cramer_critic(generated_full_1, generated_full_2) )
          # loss function for generator network (with weights)
          #generator_loss = tf.reduce_mean(cramer_critic(train_full      , generated_full_2) * w_full * w_x_2
          #                                - cramer_critic(generated_full_1, generated_full_2) * w_x_1  * w_x_2)
  
          with tf.name_scope("gradient_loss") :
            alpha             = tf.random_uniform(shape=[tf.shape(train_full)[0], 1],
                                                  minval=0., maxval=1., seed = 5678+igpu )
            interpolates      = alpha*train_full + (1.-alpha)*generated_full_1
            disc_interpolates = cramer_critic(interpolates, generated_full_2)
            gradients         = tf.gradients(disc_interpolates, [interpolates])[0]
            slopes            = tf.norm(tf.reshape(gradients, [tf.shape(gradients)[0], -1]), axis=1)
            gradient_penalty  = tf.reduce_mean(tf.square(tf.maximum(tf.abs(slopes) - 1, 0)))
    
          lambda_tf   = 20 / np.pi * 2 * tf.atan(tf.cast(g_step,tf.float32)/1e4)
          critic_loss = lambda_tf*gradient_penalty - generator_loss

          # compute and save gradients
          gpu_grads_critics.append( optimizer.compute_gradients(critic_loss) )
          gpu_grads_generators.append( optimizer.compute_gradients(generator_loss) )
  
          if args.debug :
            print("alpha\n", alpha.shape, '\n', alpha.eval() )

          # compute diff between target and trained generator output
          target_diff = generated_y_1 - train_x_1_targets
          if args.debug :
            print( "target_diff\n", target_diff.shape )

          # critic accuracy
          correct_prediction = tf.equal( tf.argmax(generated_y_1,1), tf.argmax(train_x_1_targets,1) )
          critic_accuracy    = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
          
          tf.summary.scalar("Critic_Loss",     tf.reshape(critic_loss, []))
          tf.summary.scalar("Generator_Loss",  tf.reshape(generator_loss, []))
          tf.summary.scalar("Learning_Rate",   learning_rate)
          tf.summary.scalar("Lambda",          lambda_tf)
          tf.summary.scalar("Critic_Accuracy", critic_accuracy)
          # Add histograms for generated output
          for iname in range(output_dim) :
            tf.summary.histogram( target_names[iname], generated_y_1[:,iname] )
            tf.summary.histogram( target_names[iname]+"-Diff", target_diff[:,iname] )

  # We must calculate the mean of each gradient. 
  grads_critics    = average_gradients(gpu_grads_critics)
  grads_generators = average_gradients(gpu_grads_generators)

  # Add histograms for gradients.
  for grads in [grads_critics,grads_generators] :
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
  variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,g_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  # Apply the gradients to adjust the shared variables.
  apply_gradient_op_critic = optimizer.apply_gradients( grads_critics,    global_step=g_step )
  apply_gradient_op_gen    = optimizer.apply_gradients( grads_generators, global_step=g_step )

  if args.debug : 
    print( "raw data val\n", val_raw.tail() )

  VALIDATION_SIZE   = min( args.validationsize, data_val.shape[0] )
  validation_np     = data_val.sample(VALIDATION_SIZE)
  validation_np_raw = val_raw.sample(VALIDATION_SIZE)

  if args.debug :
    print( "validation_np\n", validation_np.shape, "\n", validation_np.tail() )
    
  # number outputs from generator
  output_dim = len(target_names)

  # Output directories
  plots_dir = args.outputdir+"/"+MODEL_NAME+"/"
  dirs = RICH.outputDirs( plots_dir )
  print ( "Output dir", plots_dir )

  # Make some input / output plots
  RICH.plots1( "output_raw",  data_raw[0][target_names],   plots_dir )
  RICH.plots1( "inputs_raw",  data_raw[0][train_names],    plots_dir )
  RICH.plots1( "output_norm", data_train[0][target_names], plots_dir )
  RICH.plots1( "inputs_norm", data_train[0][train_names],  plots_dir )

  merged_summary = tf.summary.merge_all()

  var_init      = tf.global_variables_initializer()
  weights_saver = tf.train.Saver()
  tf.get_default_graph().finalize()
  
  MODEL_WEIGHTS_FILE = dirs["weights"]+"%s.ckpt" % MODEL_NAME
  train_writer       = tf.summary.FileWriter(os.path.join(dirs["summary"], "train"))
  test_writer        = tf.summary.FileWriter(os.path.join(dirs["summary"], "test"))

  # if debug, abort before starting training..
  if args.debug : sys.exit(0)

  # functor to give the number of training runs per iteration
  critic_policy = RICH.critic_policy(TOTAL_ITERATIONS)
  
  # To make sure that we can reproduce the experiment and get the same results
  RNDM_SEED = 12345
  np.random.seed(RNDM_SEED)
  tf.set_random_seed(RNDM_SEED)

with tf.Session(config=tf_config) as sess:

  # Initialise
  sess.run(var_init)
  
  # Try and restore a saved weights file
  try:
    weights_saver.restore(sess, MODEL_WEIGHTS_FILE)
    print("Restored weights from",MODEL_WEIGHTS_FILE)
  except tf.errors.NotFoundError:
    print("Can't restore parameters: no file with weights")
  
  # Do the iterations
  its = range(1,TOTAL_ITERATIONS+1)
  if not ( args.batchmode or args.debug ) : its = tqdm(its)
  for i in its :

    if args.debug : print( "Start training" )
    for j in range(critic_policy(i)) : sess.run( apply_gradient_op_critic )
    train_summary = sess.run( merged_summary )
    sess.run( [ apply_gradient_op_gen, variables_averages_op ] )
    if args.debug : print( "Finish training" )
   
    # write the summary data at given rate
    if i % args.trainwriteinterval == 0 or i == 1 :
      
      if args.debug : print( "Start train write" )
      train_writer.add_summary(train_summary)
      if args.debug : print( "Finish train write" )
   
    # Do validation now and then
    if i % args.validationinterval == 0 or i == 1 :

      if args.debug : print( "Start validation" )

      # Directory for plots etc. for this iteratons
      it_dir = dirs["iterations"]+str( '%06d' % i )+"/"
      if not os.path.exists(it_dir) : os.makedirs(it_dir)

      #clear_output(False)
      test_summary, test_generated = sess.run( [merged_summary, generated_y_1], {
          train_x_1_    : validation_np.values[:, output_dim:],
          train_x_2_    : validation_np.values[:, output_dim:],
          train_x_1_all : validation_np.values,
          train_full_   : validation_np.values } )
      
      # Summary and weights
      test_writer.add_summary(test_summary)
      weights_saver.save( sess, MODEL_WEIGHTS_FILE )
      
      # Normalised output vars
      RICH.plot2( "NormalisedDLLs", 
                  [ validation_np[target_names].values, test_generated ],
                  target_names, ['Target','Generated'], it_dir )
      
      # raw generated DLLs
      test_generated_raw = dll_scaler.inverse_transform( test_generated )
      RICH.plot2( "RawDLLs", 
                  [ validation_np_raw[target_names].values, test_generated_raw ],
                  target_names, ['Target','Generated'], it_dir )
      
      # DLL correlations
      RICH.outputCorrs( "correlations", test_generated, validation_np[target_names].values,
                        target_names, it_dir )

      if args.debug : print( "Finish validation" )
      

with tf.Session(config=tf_config) as sess:
  sess.run(var_init)
  weights_saver.restore( sess, MODEL_WEIGHTS_FILE )
  sess.graph._unsafe_unfinalize()
  tf.saved_model.simple_save(sess, dirs["model"],
                             inputs={"x": train_x_1}, outputs={"dlls": generated_y_1})
  tf.get_default_graph().finalize()
    
#from sklearn.externals import joblib
#joblib.dump(scaler, os.path.join(plots_dir, 'preprocessors', MODEL_NAME) + "_preprocessor.pkl")
