
import os, shutil
import numpy as np
import pandas as pd
import tensorflow as tf

def critic_policy(TOTAL_ITERATIONS):
    # functor to give the number of training runs per iteration
    CRITIC_ITERATIONS_CONST = 15
    CRITIC_ITERATIONS_VAR   = 0
    critic_policy = lambda i: (
        CRITIC_ITERATIONS_CONST + (CRITIC_ITERATIONS_VAR * (TOTAL_ITERATIONS - i)) // TOTAL_ITERATIONS)
    return critic_policy

def outputDirs( dir, clear = True ) :
    
    #if clear and os.path.exists(dir) : shutil.rmtree(dir) 
    if not os.path.exists(dir) : os.makedirs(dir)
    dirs = { "weights"    : dir+"weights/",
             "iterations" : dir+"iteration/",
             "summary"    : dir+"summary/",
             "model"      : dir+"exported_model/",
             "checkpoint" : dir+"checkpoint/" }
    if clear : 
        for name in dirs.keys() :
            if os.path.exists(dirs[name]) : shutil.rmtree(dirs[name]) 
    return dirs

def createRICHModel( g_step = None ) :

    import argparse, sys
    import keras
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(description='Model Parameters')
    
    parser.add_argument( '--datadir', type=str, default="/usera/jonesc/cernbox/Projects/MCGenGAN/data" )
    
    parser.add_argument( '--datareadsize', type=int, default="10000" )
    
    parser.add_argument( '--ncriticlayers', type=int, default="10" )
    parser.add_argument( '--ngeneratorlayers', type=int, default="10" )

    parser.add_argument( '--batchsize', type=int, default="1000" )
    
    parser.add_argument( '--leakrate', type=float, default="0.0" )
    parser.add_argument( '--dropoutrate', type=float, default="0.0" )
    parser.add_argument( '--cramerdim', type=int, default="256" )
    
    parser.add_argument( '--inputvars', type=str, nargs='+',
                         default = ['NumRich1Hits','NumRich2Hits','TrackP','TrackPt'] )
    
    parser.add_argument( '--outputvars', type=str, nargs='+',
                         default = [ 'RichDLLe', 'RichDLLmu', 'RichDLLk', 
                                     'RichDLLp', 'RichDLLd', 'RichDLLbt' ] )
    
    #args = parser.parse_args()
    args,unparsed = parser.parse_known_args()
    print( "Model arguments", args )

    # Job size parameters
    
    BATCH_SIZE          = args.batchsize
    maxData             = args.datareadsize
    
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
    
    # Split data into train and validation samples
    data_raw, val_raw = train_test_split( createLHCbData( train_names+target_names,  
                                                          maxData, 'KAONS',
                                                          args.datadir ),
                                          random_state = 1234 )

    # scale
    print ( "Scaling the data" )
    ndataforscale = min( 500000, data_raw.shape[0] )
    scaler     = getScaler( data_raw.iloc[0:ndataforscale] )
    dll_scaler = getScaler( data_raw[target_names].iloc[0:ndataforscale] )
    data_train = pd.DataFrame( scaler.transform(data_raw), columns = data_raw.columns, dtype=np.float32 )
    data_val   = pd.DataFrame( scaler.transform(val_raw),  columns = val_raw.columns,  dtype=np.float32 )
    print( "Training Norm Data\n", data_train.head() )

    # Build the critic and generator models

    n_input_layer = data_train.shape[1]
    print( "Building Critic, #inputs=", n_input_layer )
    critic = keras.models.Sequential()
    critic.add( keras.layers.InputLayer( [ n_input_layer ] ) )
    for i in range(0,N_LAYERS_CRITIC) :
        critic.add( keras.layers.Dense(128, activation='relu' ) )
        if LEAK_RATE    > 0 : critic.add( ll.LeakyReLU(LEAK_RATE) )
        if DROPOUT_RATE > 0 : critic.add( ll.Dropout(DROPOUT_RATE) )
    critic.add( keras.layers.Dense(CRAMER_DIM) )
    critic.summary()
        
    print( "Building Generator, #inputs=", GENERATOR_DIMENSIONS )
    generator = keras.models.Sequential()
    generator.add( keras.layers.InputLayer( [GENERATOR_DIMENSIONS] ) )
    for i in range(0,N_LAYERS_GENERATOR) :
        generator.add( keras.layers.Dense(128, activation='relu' ) )
        if LEAK_RATE    > 0 : generator.add( ll.LeakyReLU(LEAK_RATE) )
        if DROPOUT_RATE > 0 : generator.add( ll.Dropout(DROPOUT_RATE) )
    generator.add( keras.layers.Dense(output_dim) )
    generator.summary()

    # Create tensor data iterators
    print( "Creating data iterators" )
    # everything, inputs and outputs
    train_full = get_tf_dataset(data_train, BATCH_SIZE)
    # Inputs only, two sets
    train_x_1  = get_tf_dataset(data_train[train_names].values, BATCH_SIZE)
    train_x_2  = get_tf_dataset(data_train[train_names].values, BATCH_SIZE)
    
    # Create noise data
    print( "Creating noise data" )
    noise_1          = tf.random_normal([tf.shape(train_x_1)[0], NOISE_DIMENSIONS], name='noise')
    noise_2          = tf.random_normal([tf.shape(train_x_2)[0], NOISE_DIMENSIONS], name='noise')

    generated_y_1    = generator( tf.concat([noise_1, train_x_1], axis=1) )
    generated_y_2    = generator( tf.concat([noise_2, train_x_2], axis=1) )
    generated_full_1 = tf.concat([generated_y_1, train_x_1], axis=1)
    generated_full_2 = tf.concat([generated_y_2, train_x_2], axis=1)

    def cramer_critic( x, y ):
        discriminated_x = critic(x)
        return tf.norm(discriminated_x - critic(y), axis=1) - tf.norm(discriminated_x, axis=1)

    # loss function for generator network
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
    critic_train_op = optimizer.minimize( critic_loss, 
                                          var_list=critic.trainable_weights )
    gen_op          = optimizer.minimize( generator_loss,
                                          var_list=generator.trainable_weights,
                                          global_step = g_step )

    generator_train_op = tf.group( gen_op, tf.assign_add(tf_iter,1) )

    # return a dict with the various entities created
    return { "RawTrainData"       : data_raw,
             "NormTrainData"      : data_train,
             "RawValidationData"  : val_raw,
             "NormValidationData" : data_val,
             "DataScaler"         : scaler,
             "DLLScaler"          : dll_scaler,
             "TargetNames"        : target_names,
             "InputNames"         : train_names,
             "CriticLoss"         : critic_loss,
             "GeneratorLoss"      : generator_loss,
             "LearnRate"          : learning_rate,
             "TfLambda"           : lambda_tf,
             "CriticOptimizer"    : critic_train_op,
             "GeneratorOptimizer" : generator_train_op,
             "TfIterator"         : tf_iter,
             "BatchInputs"        : [ train_full, train_x_1, train_x_2 ],
             "BatchGeneratedDLLs" : [ generated_y_1, generated_y_2 ]
         }

def tfSummary( rModel ) :
    
    critic_loss    = rModel["CriticLoss"]
    generator_loss = rModel["GeneratorLoss"]
    learning_rate  = rModel["LearnRate"]
    lambda_tf      = rModel["TfLambda"]

    tf.summary.scalar("critic_loss",    tf.reshape(critic_loss, []))
    tf.summary.scalar("generator_loss", tf.reshape(generator_loss, []))
    tf.summary.scalar("learning_rate",  learning_rate)
    tf.summary.scalar("lambda",         lambda_tf)

    return tf.summary.merge_all()
    
def convertCSVtoHDF( types = ['PIONS','KAONS'] ) :

    for type in types :
        datafile = 'data/PID-train-data-'+type
        data = pd.read_csv( datafile+'.txt.gz', delim_whitespace = True )
        data = data.astype( np.float32 )
        data.to_hdf( datafile+'.hdf', type, mode='w', complib='blosc')

# load the LHCb data
def createLHCbData( names, maxData = -1, type = 'KAONS', 
                    datapath = '/usera/jonesc/cernbox/Projects/MCGenGAN/data',
                    dtype = np.float32 ) :

    datafile = datapath+'/PID-train-data-'+type +'.hdf'
    
    print( "Loading the data from", datafile )
    data = pd.read_hdf( datafile, type )

    if maxData < 0 or data.shape[0] < maxData :
        return data[names].astype(dtype)
    else:
        return data[names].iloc[0:maxData].astype(dtype)

def getScaler( data ):
    from sklearn.preprocessing import QuantileTransformer
    return QuantileTransformer(output_distribution="normal",
                                 n_quantiles=int(1e5),
                                 subsample=int(1e10)).fit(data)

def plotDimensions( nvars ):
    # work out the dimensions to use when plotting
    import math
    iy = ix = int(math.sqrt(nvars))
    if ix*iy < nvars : ix = ix+1
    if ix*iy < nvars : iy = iy+1
    return (ix,iy)

def plots1( title, data, dir = 'plots/' ):

    import matplotlib.pyplot as plt
    import os

    nvars = len(data.columns)
    if nvars > 0 :

        print( "Producing", title, "plots in", dir )
    
        if not os.path.exists(dir) : os.makedirs(dir)

        ix, iy = plotDimensions(nvars)

        plt.figure(figsize=(18,15))
        i = 0
        for col in data.columns :
  
            plt.subplot(ix, iy, i+1)
            plt.hist( data[col], 100, histtype='bar' )
            plt.grid(True)
            plt.title( col )
            i = i + 1

        plt.tight_layout()
        plt.savefig(dir+title+'.png')
        plt.close()

def plot2( title, data, names, label=['Target','Generated'], dir = 'plots/' ):

    import matplotlib.pyplot as plt

    # plot dimensions
    ix,iy = plotDimensions(len(names))
    
    plt.figure(figsize=(18,15))
    for INDEX in range(0,len(names)) :
        plt.subplot(ix, iy, INDEX+1)
        d = [ d[:, INDEX] for d in data ]
        plt.hist( d, bins=100, alpha=0.5, density=True, 
                  histtype='stepfilled', label=label ) 
        plt.grid(True)
        plt.title(title+" "+names[INDEX])
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir+title+".png")
    plt.close()


def initPlots( rModel, plots_dir ) :

    data_raw       = rModel["RawTrainData"]
    data_train     = rModel["NormTrainData"]
    train_names    = rModel["InputNames"]
    target_names   = rModel["TargetNames"]

    # Make some input / output plots
    plots1( "output_raw",  data_raw[target_names],   plots_dir )
    plots1( "inputs_raw",  data_raw[train_names],    plots_dir )
    plots1( "output_norm", data_train[target_names], plots_dir )
    plots1( "inputs_norm", data_train[train_names],  plots_dir )

def outputCorrs( title, generated_out, true_out, names, dir = 'plots/' ):

    import matplotlib.pyplot as plt

    nvars = len(names)
    if nvars > 0 :

        ix, iy = plotDimensions(nvars)

        plt.figure(figsize=(15,15))
        for i in range(0,len(names)) :
            
            plt.subplot(ix, iy, i+1)
            plt.hist2d( generated_out[:,i], true_out[:,i],
                        range = [[-3,3],[-3,3]], bins = 50 )
            plt.grid(True)
            plt.title( names[i] )
            plt.xlabel('Generated')
            plt.ylabel('Target')
            plt.colorbar()
    
        plt.tight_layout()
        plt.savefig(dir+title+'.png')
        plt.close()

def get_tf_dataset( dataset, batch_size, dtype = np.float32 ):
    import tensorflow as tf
    #shuffler   = tf.data.experimental.shuffle_and_repeat(dataset.shape[0])
    shuffler   = tf.contrib.data.shuffle_and_repeat(dataset.shape[0])
    suffled_ds = shuffler(tf.data.Dataset.from_tensor_slices(dataset))
    return suffled_ds.batch(batch_size).prefetch(1).make_one_shot_iterator().get_next()
    #return tf.data.Dataset.from_tensor_slices(dataset).batch(batch_size).prefetch(1).make_one_shot_iterator().get_next()
