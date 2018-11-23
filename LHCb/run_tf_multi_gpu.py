#! /usr/bin/env python3

import os, sys, platform, argparse
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from tqdm import tqdm
import RICH
import keras
from sklearn.model_selection import train_test_split
import pandas as pd

print( "Running on", platform.node() )

parser = argparse.ArgumentParser(description='Job Parameters')

parser.add_argument( '--name', type=str, default="" )

parser.add_argument( '--outputdir', type=str, default="/home/jonesc/Projects/RICHMCGenGan/output" )
parser.add_argument( '--datadir', type=str, default="/home/jonesc/Projects/RICHMCGenGan/data" )

parser.add_argument( '--datareadsize', type=int, default="4000000" )

parser.add_argument( '--niterations', type=int, default="100000" )

parser.add_argument( '--validationsize', type=int, default="300000" )
parser.add_argument( '--valfrac', type=float, default="0.2" )

parser.add_argument( '--batchsize', type=int, default="50000" )

parser.add_argument( '--validationinterval', type=int, default="100" )
parser.add_argument( '--trainwriteinterval', type=int, default="50" )

parser.add_argument( '--ngpus', type=int, default="3" )
parser.add_argument( '--gpumergeinterval', type=int, default="100" )

parser.add_argument( '--noisedims', type=int, default="64" )

parser.add_argument( '--ncriticlayers', type=int, default="10" )
parser.add_argument( '--criticinnerdim', type=int, default="128" )
parser.add_argument( '--criticleakrate', type=float, default="0.0" )
parser.add_argument( '--criticdropoutrate', type=float, default="0.0" )
parser.add_argument( '--cramerdim', type=int, default="256" )

parser.add_argument( '--ngeneratorlayers', type=int, default="10" )
parser.add_argument( '--generatorinnerdim', type=int, default="128" )
parser.add_argument( '--generatorleakrate', type=float, default="0.0" )
parser.add_argument( '--generatordropoutrate', type=float, default="0.0" )

parser.add_argument( '--begincritictrainperit', type=int, default="10" )
parser.add_argument( '--endcritictrainperit',   type=int, default="20" )

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

parser.add_argument( '--debug', action='store_true' )
parser.set_defaults(debug=False)

args,unparsed = parser.parse_known_args()
print( "Job arguments", args )

# Build full name from options
MODEL_NAME = ""
if args.name != "" : MODEL_NAME = args.name + "-"
name_map = {  'TrackP'           : 'P'
             ,'TrackPt'          : 'Pt'
             ,'NumRich1Hits'     : 'R1H'
             ,'NumRich2Hits'     : 'R2H'
             ,'TrackRich1EntryX' : 'R1EnX'
             ,'TrackRich1EntryY' : 'R1EnY'
             ,'TrackRich1ExitX'  : 'R1ExX'
             ,'TrackRich1ExitY'  : 'R1ExY'
             ,'TrackRich2EntryX' : 'R2EnX'
             ,'TrackRich2EntryY' : 'R2EnY'
             ,'TrackRich2ExitX'  : 'R2ExX'
             ,'TrackRich2ExitY'  : 'R2ExY'
           }
for i in args.inputvars : MODEL_NAME += name_map[i]
if args.criticdropoutrate    > 0 : MODEL_NAME += "-CrtDrop"+str(args.criticdropoutrate)
if args.criticleakrate       > 0 : MODEL_NAME += "-CrtLeak"+str(args.criticleakrate)
MODEL_NAME += "-CriInnerD" + str(args.criticinnerdim)
MODEL_NAME += "-CriNL"  + str(args.ncriticlayers)
MODEL_NAME += "-CramerD"+ str(args.cramerdim)
MODEL_NAME += "-NoiseD" + str(args.noisedims)
if args.generatordropoutrate > 0 : MODEL_NAME += "-GenDrop"+str(args.generatordropoutrate)
if args.generatorleakrate    > 0 : MODEL_NAME += "-GenLeak"+str(args.generatorleakrate)
MODEL_NAME += "-GenInnerD" + str(args.generatorinnerdim)
MODEL_NAME += "-GenNL"  + str(args.ngeneratorlayers)
print( "Model Name", MODEL_NAME )

# first line must be before tensorflow import
if args.debug : os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
tf.reset_default_graph()

# debugging
if args.debug : sess = tf.InteractiveSession()

# To make sure that we can reproduce the experiment and get the same results
RNDM_SEED = 12345
np.random.seed(RNDM_SEED)
tf.set_random_seed(RNDM_SEED)

# number outputs from generator
output_dim = len(args.outputvars)

with tf.device('/cpu:0'):

  # global step 
  g_step = tf.train.get_or_create_global_step()
  # op to increment the global step
  increment_global_step_op = tf.assign(g_step, g_step+1)

  # read data from file
  readData = RICH.createLHCbData( args.outputvars + args.inputvars,  
                                  args.datareadsize, 'KAONS',
                                  args.datadir )

  # create data scaler
  ndataforscale = min( 1000000, readData.shape[0] )
  scaler     = RICH.getScaler( readData.iloc[0:ndataforscale] )
  #scaler     = RICH.getScaler( readData )
  dll_scaler = RICH.getScaler( readData[args.outputvars].iloc[0:ndataforscale] )

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

  # Save critic and generator op calls
  gpu_gen_ops    = [ ]
  gpu_critic_ops = [ ]
  # save the generator and critics
  gpu_critics    = [ ]
  gpu_generators = [ ]
  # accuracy ops
  gen_acc_ops    = [ ]

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
          for i in range(0,args.ncriticlayers) :
            critic_i.add( keras.layers.Dense(args.criticinnerdim, activation='relu' ) )
            if args.criticleakrate    > 0 : 
              critic_i.add( keras.layers.LeakyReLU(args.criticleakrate) )
            if args.criticdropoutrate > 0 : 
              critic_i.add( keras.layers.Dropout(args.criticdropoutrate) )
          critic_i.add( keras.layers.Dense(args.cramerdim) )
          if args.debug : critic_i.summary()
          gpu_critics.append(critic_i)
    
          # Total input dimensions of generator (including noise)
          GENERATOR_DIMENSIONS = args.noisedims + len(args.inputvars)
          print( "Building Generator, #inputs=", GENERATOR_DIMENSIONS )
          generator_i = keras.models.Sequential()
          generator_i.add( keras.layers.InputLayer( [GENERATOR_DIMENSIONS] ) )
          for i in range(0,args.ngeneratorlayers) :
            generator_i.add( keras.layers.Dense(args.generatorinnerdim, activation='relu' ) )
            if args.generatorleakrate    > 0 : 
              generator_i.add( keras.layers.LeakyReLU(args.generatorleakrate) )
            if args.generatordropoutrate > 0 : 
              generator_i.add( keras.layers.Dropout(args.generatordropoutrate) )
          generator_i.add( keras.layers.Dense(output_dim) )
          if args.debug : generator_i.summary()
          gpu_generators.append(generator_i)
     
          # Create tensor data iterators
          print( "Creating data iterators" )

          # reference data sample
          train_ref = RICH.get_tf_dataset( data_train[igpu], args.batchsize )

          # lists of various views of the data, for independent samples
          train_full       = [ ] # Primary data iterator. Target DLls and physics input. 
                                 # input to critic
          train_phys       = [ ] # physics input data to generator
          target_dlls      = [ ] # target DLLs values
          train_noise      = [ ] # noise input data to geneator
          generator_inputs = [ ] # All inputs to the generator
          generated_dlls   = [ ] # Generated DLL values
          gen_critic_input = [ ] # Input to the critic for the generated DLLs

          # create the data views for two independent samples
          for sample in range(2) :
  
            # full data random iterator
            train_full.append( RICH.get_tf_dataset( data_train[igpu], args.batchsize ) )

            # physics inputs to the networks
            train_phys.append( train_full[-1][:,output_dim:] ) 

            # target DLL values
            target_dlls.append( train_full[-1][:,:output_dim] ) 

            # input noise data
            train_noise.append( tf.random_normal( [tf.shape(train_phys[-1])[0], args.noisedims], 
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
          generator_loss = tf.reduce_mean( cramer_critic( train_ref          , gen_critic_input[1]) 
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

          # generator accuracy
          gen_acc, gen_acc_op = tf.metrics.accuracy( tf.argmax(generated_dlls[0],1) , 
                                                     tf.argmax(target_dlls[0]   ,1) )
          gen_acc_ops.append( gen_acc_op )
          dll_equal = tf.equal( tf.argmax(generated_dlls[0],1) , 
                                tf.argmax(target_dlls[0]   ,1) )
          gen_acc_2 = tf.reduce_mean( tf.cast(dll_equal,tf.float32) )
          # critic accuracy
          cramer_equal = tf.equal( tf.argmax( critic_i(train_full[0])       , 1 ) ,
                                   tf.argmax( critic_i(gen_critic_input[0]) , 1 ) )
          critic_acc = tf.reduce_mean( tf.cast(cramer_equal,tf.float32) )
          
          tf.summary.scalar("Critic_Loss",        tf.reshape(critic_loss,[]) )
          tf.summary.scalar("Generator_Loss",     tf.reshape(generator_loss,[]) )
          tf.summary.scalar("Learning_Rate",      learning_rate )
          tf.summary.scalar("Lambda",             lambda_tf )
          tf.summary.scalar("Generator_Accuracy_1", gen_acc )
          tf.summary.scalar("Generator_Accuracy_2", gen_acc_2 )
          tf.summary.scalar("Critic_Accuracy",      critic_acc )
          # Add histograms for generated output
          for iname in range(output_dim) :
            tf.summary.histogram( args.outputvars[iname], generated_dlls[0][:,iname] )
            tf.summary.histogram( args.outputvars[iname]+"-Diff", target_diff[:,iname] )

  # list over weight merging ops
  weight_merge_ops = [ ]
  i_w_gpu = 0
  
  # loop over the lists of critic and generator models
  for models in [ gpu_critics, gpu_generators ] : 

    # loop over the shared weights for each model in this list
    # All have the same structure so use the first entry to get the number.
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
  del data_val
  del val_raw

  if args.debug :
    print( "validation_np\n", validation_np[0].shape, "\n", validation_np[0].tail() )
    
  # number outputs from generator
  output_dim = len(args.outputvars)

  # Output directories
  plots_dir = args.outputdir+"/"+MODEL_NAME+"/"
  dirs = RICH.outputDirs( plots_dir )
  print ( "Output dir", plots_dir )

  # Make some input / output plots
  RICH.plots1( "output_raw ",  data_raw[0][args.outputvars],   plots_dir )
  RICH.plots1( "inputs_raw ",  data_raw[0][args.inputvars],    plots_dir )
  RICH.plots1( "output_norm", data_train[0][args.outputvars], plots_dir )
  RICH.plots1( "inputs_norm", data_train[0][args.inputvars],  plots_dir )

  # Summary and weights writers
  merged_summary     = tf.summary.merge_all()
  var_init_glo       = tf.global_variables_initializer()
  var_init_loc       = tf.local_variables_initializer()
  weights_saver      = tf.train.Saver( name = args.name, max_to_keep = 3 )  
  MODEL_WEIGHTS_FILE = dirs["weights"]+"model-weights.ckpt"
  train_writer       = tf.summary.FileWriter(os.path.join(dirs["summary"],"train"))
  test_writer        = tf.summary.FileWriter(os.path.join(dirs["summary"],"test"))

  # Configuration
  tf_config = tf.ConfigProto( gpu_options = tf.GPUOptions(allow_growth=False),
                              allow_soft_placement = True,
                              log_device_placement = args.debug,
                              intra_op_parallelism_threads = 16,
                              inter_op_parallelism_threads = 16 )

  # Finalise the graph for the run
  tf.get_default_graph().finalize()

  # if debug, abort before starting training..
  if args.debug : sys.exit(0)

  # functor to give the number of training runs per iteration
  def create_critic_policy():
    scale_f = ( args.endcritictrainperit - args.begincritictrainperit ) / args.niterations
    critic_policy = lambda i: int( round( args.begincritictrainperit + ( scale_f * (i-1) ) ) )
    return critic_policy
  critic_policy = create_critic_policy()

# Start the session ....
with tf.Session(config=tf_config) as sess:

  # Initialise
  sess.run(var_init_loc)
  sess.run(var_init_glo)

  # Try and restore a saved weights file
  try:
    weights_saver.restore( sess, MODEL_WEIGHTS_FILE )
    print("Restored weights from",MODEL_WEIGHTS_FILE)
  except tf.errors.NotFoundError:
    print("Weights file not found from",MODEL_WEIGHTS_FILE)
  
  # Do the iterations
  its = range(1,args.niterations+1)
  if not args.debug : its = tqdm(its)
  for i in its :

    # read the global step count
    g_it = sess.run( g_step )

    # skip run iterations
    if i-1 < g_it : continue

    if args.debug : print( "Start training" )
    for j in range(critic_policy(i)) : sess.run( gpu_critic_ops )
    sess.run( gpu_gen_ops + gen_acc_ops + [ increment_global_step_op ] )
    train_summary = sess.run( merged_summary )
    if args.debug : print( "Finish training" )
   
    # write the summary data at given rate
    if i % args.trainwriteinterval == 0 :
      
      if args.debug : print( "Start train write" )
      train_writer.add_summary( train_summary, g_it )
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

      test_summary, test_generated, _, _, _ = sess.run( [ merged_summary,
                                                          generated_dlls[0],
                                                          critic_acc, 
                                                          critic_loss,
                                                          generator_loss ], {
        train_full[0]  : validation_np[0].values,
        train_full[1]  : validation_np[1].values,
        train_phys[0]  : validation_np[0].values[:,output_dim:],
        train_phys[1]  : validation_np[1].values[:,output_dim:],
        target_dlls[0] : validation_np[0].values[:,:output_dim],
        target_dlls[1] : validation_np[1].values[:,:output_dim],
        train_ref      : validation_np[2].values } )

      # Summary and weights
      test_writer.add_summary( test_summary, g_it )
      weights_saver.save( sess, MODEL_WEIGHTS_FILE )
      
      # Normalised output vars
      RICH.plot2( "NormalisedDLLs", 
                  [ validation_np[0][args.outputvars].values, test_generated ],
                  args.outputvars, ['Target','Generated'], it_dir )
      
      # raw generated DLLs
      test_generated_raw = dll_scaler.inverse_transform( test_generated )
      RICH.plot2( "RawDLLs", 
                  [ validation_np_raw[args.outputvars].values, test_generated_raw ],
                  args.outputvars, ['Target','Generated'], it_dir )
      
      # DLL correlations
      RICH.outputCorrs( "correlations", test_generated, validation_np[0][args.outputvars].values,
                        args.outputvars, it_dir )

      if args.debug : print( "Finish validation" )

with tf.Session(config=tf_config) as sess:
  sess.run(var_init_glo)
  weights_saver.restore( sess, MODEL_WEIGHTS_FILE )
  sess.graph._unsafe_unfinalize()
  tf.saved_model.simple_save(sess, dirs["model"],
                             inputs={"x": train_phys[0]}, outputs={"dlls": generated_dlls[0]})
  tf.get_default_graph().finalize()
    
#from sklearn.externals import joblib
#joblib.dump(scaler, os.path.join(plots_dir, 'preprocessors', MODEL_NAME) + "_preprocessor.pkl")
