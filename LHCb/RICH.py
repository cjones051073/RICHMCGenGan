
import os, shutil
import numpy as np
import pandas as pd
import tensorflow as tf

def critic_policy(TOTAL_ITERATIONS):
    # functor to give the number of training runs per iteration
    CRITIC_ITERATIONS_CONST = 15
    CRITIC_ITERATIONS_VAR   = 0
    critic_policy = lambda i: (
        CRITIC_ITERATIONS_CONST + (CRITIC_ITERATIONS_VAR * (TOTAL_ITERATIONS - i)) // TOTAL_ITERATIONS )
    return critic_policy

def splitDataFrame( data, nsplits ) :
    splits = np.array_split( data, nsplits )
    return [ pd.DataFrame( s, columns = data.columns, dtype=np.float32 ) for s in splits ]

def outputDirs( dir, clear = True ) :

    #if clear and os.path.exists(dir) : shutil.rmtree(dir) 
    if not os.path.exists(dir) : os.makedirs(dir)
    dirs = { "weights"    : dir+"weights/",
             "iterations" : dir+"iteration/",
             "summary"    : dir+"summary/",
             "model"      : dir+"model/",
             "checkpoint" : dir+"checkpoint/" }
    if clear : 
        for name in [ "model" ] :
            if os.path.exists(dirs[name]) : shutil.rmtree(dirs[name]) 
    return dirs

def convertCSVtoHDF( infile, outfile ) :

    data = pd.read_csv( infile, delim_whitespace = True )
    data = data.astype( np.float32 )
    data.to_hdf( outfile, type, mode='w', complib='blosc')

# load the LHCb data
def createLHCbData( names, maxData = -1, type = 'KAONS', 
                    datapath = '/usera/jonesc/cernbox/Projects/MCGenGAN/data',
                    dtype = np.float32 ) :

    datafile = datapath+'/PID-train-data-'+type +'.hdf'

    print( "Loading the data from", datafile )
    data = pd.read_hdf( datafile, type )

    # add missing weights column. In future might be useful but for now just 1.0
    #data[weight_col] = 1.0

    if maxData < 0 or data.shape[0] < maxData :
        return data[names].astype(dtype)
    else:
        return data[names].iloc[0:maxData].astype(dtype)

def getScaler( data ):
    from sklearn.preprocessing import QuantileTransformer
    return QuantileTransformer(output_distribution="normal",
                               n_quantiles=int(1e5),
                               subsample=int(1e10)).fit(data.values)

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

        ix,iy = plotDimensions(nvars)

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

    # randomise the data
    shuffler   = tf.contrib.data.shuffle_and_repeat(dataset.shape[0])
    suffled_ds = shuffler(tf.data.Dataset.from_tensor_slices(dataset))

    # return iterator for a single pass through
    return suffled_ds.batch(batch_size).prefetch(1).make_one_shot_iterator().get_next()
