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

parser.add_argument( '--name', type=str, default="PPtR1R2HitsR12EntryExit-NEW" )

parser.add_argument( '--outputdir', type=str, default="/home/jonesc/Projects/RICHMCGenGan/output" )

parser.add_argument( '--datareadsize', type=int, default="4000000" )

parser.add_argument( '--niterations', type=int, default="100000" )

parser.add_argument( '--validationsize', type=int, default="300000" )
parser.add_argument( '--valfrac', type=float, default="0.2" )

parser.add_argument( '--batchsize', type=int, default="50000" )

parser.add_argument( '--validationinterval', type=int, default="100" )
parser.add_argument( '--trainwriteinterval', type=int, default="50" )

parser.add_argument( '--ngpus', type=int, default="3" )
parser.add_argument( '--gpumergeinterval', type=int, default="100" )

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
                              allow_soft_placement = False )
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

with tf.device('/cpu:0'):

  # global step 
  g_step = tf.train.get_or_create_global_step()
  # op to increment the global step
  increment_global_step_op = tf.assign(g_step, g_step+1)

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
  data_val = pd.DataFrame( scaler.transform(val_raw), columns = val_raw.columns, dtype=np.float32 )
                                         
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
  #weight_col = 'Weight'
  #data_val[weight_col] = np.float32(1.0)
  #for d in data_train : d[weight_col] = np.float32(1.0)

  # Save critic and generator op calls
  gpu_gen_ops    = [ ]
  gpu_critic_ops = [ ]
  # save the generator and critics
  gpu_critics    = [ ]
  gpu_generators = [ ]

  with tf.variable_scope(tf.get_variable_scope()):
    for igpu in range(args.ngpus):
      tfdevice = '/gpu:%d' % igpu
      if args.debug : tfdevice = '/cpu:0'
      with tf.device(tfdevice):
        with tf.name_scope( '%s_%d' % ("GPU",igpu) ) as scope:

          print( "Configuring", tfdevice, "data sample size", data_train[igpu].shape[0] )

          # Build the critic and generator models

          #n_input_layer = data_train[igpu].shape[1] - 1 # subtract one as weights not used
          n_input_layer = data_train[igpu].shape[1]
          print( "Building Critic, #inputs=", n_input_layer )
          critic_i = keras.models.Sequential()
          critic_i.add( keras.layers.InputLayer( [ n_input_layer ] ) )
          for i in range(0,N_LAYERS_CRITIC) :
            critic_i.add( keras.layers.Dense(128, activation='relu' ) )
            if LEAK_RATE    > 0 : critic_i.add( keras.layers.LeakyReLU(LEAK_RATE) )
            if DROPOUT_RATE > 0 : critic_i.add( keras.layers.Dropout(DROPOUT_RATE) )
          critic_i.add( keras.layers.Dense(CRAMER_DIM) )
          critic_i.summary()
          gpu_critics.append(critic_i)
    
          print( "Building Generator, #inputs=", GENERATOR_DIMENSIONS )
          generator_i = keras.models.Sequential()
          generator_i.add( keras.layers.InputLayer( [GENERATOR_DIMENSIONS] ) )
          for i in range(0,N_LAYERS_GENERATOR) :
            generator_i.add( keras.layers.Dense(128, activation='relu' ) )
            if LEAK_RATE    > 0 : generator_i.add( keras.layers.LeakyReLU(LEAK_RATE) )
            #if DROPOUT_RATE > 0 : generator_i.add( keras.layers.Dropout(DROPOUT_RATE) )
          generator_i.add( keras.layers.Dense(output_dim) )
          generator_i.summary()
          gpu_generators.append(generator_i)
     
          # Create tensor data iterators
          print( "Creating data iterators" )

          # lists of various views of the data, for independent samples
          train_full       = [ ] # Primary data iterator. Target DLls and physics input. 
                                 # input to critic
          train_phys       = [ ] # physics input data to generator
          target_dlls      = [ ] # target DLLs values
          train_noise      = [ ] # noise input data to geneator
          generator_inputs = [ ] # All inputs to the generator
          generated_dlls   = [ ] # Generated DLL values
          gen_critic_input = [ ] # Input to the critic for the generated DLLs

          # reference data sample
          train_ref = RICH.get_tf_dataset( data_train[igpu], BATCH_SIZE )

          # create the data views for two independent samples
          for sample in range(2) :
  
            # full data random iterator
            train_full.append( RICH.get_tf_dataset( data_train[igpu], BATCH_SIZE ) )

            # physics inputs to the networks
            train_phys.append( train_full[-1][:,output_dim:] ) 

            # target DLL values
            target_dlls.append( train_full[-1][:,:output_dim] ) 

            # input noise data
            train_noise.append( tf.random_normal( [tf.shape(train_phys[-1])[0], NOISE_DIMENSIONS], 
                                                  name='noise'+str(sample) ) )

            # complete inputs to generator
            generator_inputs.append( tf.concat([train_noise[-1],train_phys[-1]], axis=1) )

            # generated output
            generated_dlls.append( generator_i( generator_inputs[-1] ) )

            # generated input to critic
            gen_critic_input.append( tf.concat([generated_dlls[-1],train_phys[-1]], axis=1) )

            if args.debug :
              print( "Sample", sample )
              print( "All data", train_full[-1].shape, "\n", train_full[-1].eval() )
              print( "phys input", train_phys[-1].shape, "\n", train_phys[-1].eval() )
              print( "target dlls", target_dlls[-1].shape, "\n", target_dlls[-1].eval() )
              print( "noise input", train_noise[-1].shape, "\n", train_noise[-1].eval() )
              print( "generator input", generator_inputs[-1].shape )
              print( "generated dlls", generated_dlls[-1].shape )
              print( "gen critic input", gen_critic_input[-1].shape )

          def cramer_critic( x, y ):
            discriminated_x = critic_i(x)
            discriminated_y = critic_i(y)
            return tf.norm( discriminated_x - discriminated_y, axis=1 ) - tf.norm( discriminated_x, axis=1 )
  
          # loss function for generator network (when weights are all 1)
          generator_loss = tf.reduce_mean(cramer_critic( train_ref          , gen_critic_input[1]) 
                                        - cramer_critic( gen_critic_input[0], gen_critic_input[1]) )
  
          with tf.name_scope("gradient_loss") :
            alpha             = tf.random_uniform(shape=[tf.shape(train_ref)[0], 1], minval=0., maxval=1. )
            interpolates      = alpha*train_ref + (1.-alpha)*gen_critic_input[0]
            disc_interpolates = cramer_critic( interpolates, gen_critic_input[1] )
            gradients         = tf.gradients(disc_interpolates, [interpolates])[0]
            slopes            = tf.norm(tf.reshape(gradients, [tf.shape(gradients)[0], -1]), axis=1)
            gradient_penalty  = tf.reduce_mean(tf.square(tf.maximum(tf.abs(slopes) - 1, 0)))
    
          tf_iter         = tf.Variable(initial_value=0, dtype=tf.int32)
          lambda_tf       = 20 / np.pi * 2 * tf.atan(tf.cast(tf_iter,tf.float32)/1e4)
          critic_loss     = lambda_tf*gradient_penalty - generator_loss
          learning_rate   = tf.train.exponential_decay( 2e-3, tf_iter, 200, 0.985 )
          
          optimizer       = tf.train.RMSPropOptimizer(learning_rate)
          #optimizer       = tf.train.AdamOptimizer(learning_rate)

          critic_train_op_i = optimizer.minimize( critic_loss, 
                                                  var_list=critic_i.trainable_weights )
          gpu_critic_ops.append( critic_train_op_i )
          
          gen_train_op_i = tf.group( optimizer.minimize( generator_loss,
                                                         var_list=generator_i.trainable_weights ),
                                     tf.assign_add(tf_iter,1) )
          gpu_gen_ops.append( gen_train_op_i )
  
          if args.debug :
            print("alpha\n", alpha.shape, '\n', alpha.eval() )
            print("interpolates", interpolates.shape )
            print("gradients", gradients.shape )
            print("slopes", slopes.shape )
            print("gradient_penalty", gradient_penalty.shape )

          # compute diff between target and trained generator output
          target_diff = generated_dlls[0] - target_dlls[0]
          if args.debug :
            print( "target_diff\n", target_diff.shape )

          # critic accuracy
          correct_prediction = tf.equal( tf.argmax(generated_dlls[0],1), tf.argmax(target_dlls[0],1) )
          critic_accuracy    = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
          
          tf.summary.scalar("Critic_Loss",     tf.reshape(critic_loss, []))
          tf.summary.scalar("Generator_Loss",  tf.reshape(generator_loss, []))
          tf.summary.scalar("Learning_Rate",   learning_rate)
          tf.summary.scalar("Lambda",          lambda_tf)
          tf.summary.scalar("Critic_Accuracy", critic_accuracy)
          # Add histograms for generated output
          for iname in range(output_dim) :
            tf.summary.histogram( target_names[iname], generated_dlls[0][:,iname] )
            tf.summary.histogram( target_names[iname]+"-Diff", target_diff[:,iname] )

  # list over weight merging ops
  weight_merge_ops = [ ]
  i_w_gpu = 0
  
  # loop over the lists of critic and generator models
  for models in [ gpu_critics, gpu_generators ] : 

    # loop over the shared weights for each model in this list
    # assume all the same so use the [0] entry.
    for iw in range( len(models[0].trainable_weights) ) :

      # choose the GPU for this op
      tfdevice = '/gpu:%d' % i_w_gpu
      if args.debug : tfdevice = '/cpu:0'
      with tf.device(tfdevice):

        # create average weight
        w = models[0].trainable_weights[iw]
        for ii in range(1,args.ngpus) :
          w = tf.math.add( w, models[ii].trainable_weights[iw] )
        w /= np.float32(args.ngpus)

        # assign to each model
        for ii in range(0,args.ngpus) :
          op = tf.assign( models[ii].trainable_weights[iw], w )
          weight_merge_ops.append(op)

      # pick the GPU for the next op
      i_w_gpu += 1
      if i_w_gpu == args.ngpus : i_w_gpu = 0

  if args.debug : 
    print( "raw data val\n", val_raw.tail() )

  VALIDATION_SIZE   = min( args.validationsize, data_val.shape[0] )
  # make three independent samples
  validation_np     = [ data_val.sample(VALIDATION_SIZE),
                        data_val.sample(VALIDATION_SIZE),
                        data_val.sample(VALIDATION_SIZE) ]
  validation_np_raw = val_raw.sample(VALIDATION_SIZE)

  if args.debug :
    print( "validation_np\n", validation_np[0].shape, "\n", validation_np[0].tail() )
    
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
  train_writer       = tf.summary.FileWriter(os.path.join(dirs["summary"],"train"))
  test_writer        = tf.summary.FileWriter(os.path.join(dirs["summary"],"test"))

  # if debug, abort before starting training..
  if args.debug : sys.exit(0)

  # functor to give the number of training runs per iteration
  critic_policy = RICH.critic_policy(TOTAL_ITERATIONS)

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

    # read the global step count
    g_it = sess.run( g_step )

    # skip run iterations
    if i-1 < g_it : continue

    if args.debug : print( "Start training" )
    for j in range(critic_policy(i)) : sess.run( gpu_critic_ops )
    train_summary = sess.run( merged_summary )
    sess.run( gpu_gen_ops )
    sess.run( increment_global_step_op )
    if args.debug : print( "Finish training" )
   
    # write the summary data at given rate
    if i % args.trainwriteinterval == 0 :
      
      if args.debug : print( "Start train write" )
      train_writer.add_summary(train_summary,g_it)
      if args.debug : print( "Finish train write" )
   
    # merge the GPU info at a given rate
    if i % args.gpumergeinterval == 0 :

      if args.debug : print( "Start GPU merge" )
      sess.run( weight_merge_ops )
      if args.debug : print( "Finish GPU merge" )

    # Do validation now and then
    if i % args.validationinterval == 0 :

      if args.debug : print( "Start validation" )

      # Directory for plots etc. for this iteratons
      it_dir = dirs["iterations"]+str( '%06d' % i )+"/"
      if not os.path.exists(it_dir) : os.makedirs(it_dir)

      test_summary, test_generated = sess.run( [merged_summary,generated_dlls[0]], {
         train_full[0] : validation_np[0].values,
         train_full[1] : validation_np[1].values,
         train_ref     : validation_np[2].values } )
      
      # Summary and weights
      test_writer.add_summary(test_summary, g_it )
      weights_saver.save( sess, MODEL_WEIGHTS_FILE, global_step = g_step )
      
      # Normalised output vars
      RICH.plot2( "NormalisedDLLs", 
                  [ validation_np[0][target_names].values, test_generated ],
                  target_names, ['Target','Generated'], it_dir )
      
      # raw generated DLLs
      test_generated_raw = dll_scaler.inverse_transform( test_generated )
      RICH.plot2( "RawDLLs", 
                  [ validation_np_raw[target_names].values, test_generated_raw ],
                  target_names, ['Target','Generated'], it_dir )
      
      # DLL correlations
      RICH.outputCorrs( "correlations", test_generated, validation_np[0][target_names].values,
                        target_names, it_dir )

      if args.debug : print( "Finish validation" )
      

with tf.Session(config=tf_config) as sess:
  sess.run(var_init)
  weights_saver.restore( sess, MODEL_WEIGHTS_FILE )
  sess.graph._unsafe_unfinalize()
  tf.saved_model.simple_save(sess, dirs["model"],
                             inputs={"x": train_phys[0]}, outputs={"dlls": generated_dlls[0]})
  tf.get_default_graph().finalize()
    
#from sklearn.externals import joblib
#joblib.dump(scaler, os.path.join(plots_dir, 'preprocessors', MODEL_NAME) + "_preprocessor.pkl")
