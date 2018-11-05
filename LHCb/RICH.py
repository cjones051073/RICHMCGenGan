
import os, shutil
import numpy  as np
import pandas as pd

def convertCVStoHDF( types = ['PIONS','KAONS'] ) :

    for type in types :
        datafile = 'data/PID-train-data-'+type
        data = pd.read_csv( datafile+'.txt.gz', delim_whitespace=True )
        data = data.astype( np.float32 )
        data.to_hdf( datafile+'.hdf', type, mode='w', complib='blosc')

# load the LHCb data
def createLHCbData( names, maxData = -1, type = 'KAONS', 
                    datapath = '/usera/jonesc/NFS/data/MCGenGAN',
                    dtype = np.float32 ):

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

def plots( title, data, dir = 'plots/' ):

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
            #print( "Plotting", i, names[i] )
            plt.subplot(ix, iy, i+1)
            plt.hist( data[col], 50, histtype='bar' )
            plt.grid(True)
            plt.title( col )
            i = i + 1

        plt.tight_layout()
        plt.savefig(dir+title+'.png')
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
    shuffler   = tf.contrib.data.shuffle_and_repeat(dataset.shape[0])
    suffled_ds = shuffler(tf.data.Dataset.from_tensor_slices(dataset))
    return suffled_ds.batch(batch_size).prefetch(1).make_one_shot_iterator().get_next()
