#! /usr/bin/env python3

#from sklearn.preprocessing import QuantileTransformer
import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os, shutil, sys, platform
from tqdm import tqdm
import RICH
import argparse

print( "Running on", platform.node() )

parser = argparse.ArgumentParser(description='Job Parameters')

parser.add_argument( '--name', type=str, default="Test1" )

parser.add_argument( '--outputdir', type=str, default="/usera/jonesc/NFS/output/MCGenGAN" )

parser.add_argument( '--validationsize', type=int, default="100" )
parser.add_argument( '--validationinterval', type=int, default="10" )
parser.add_argument( '--niterations', type=int, default="100" )

parser.add_argument( '--batchmode', action='store_true' )
parser.set_defaults(batchmode=False)

args,unparsed = parser.parse_known_args()
print( "Job arguments", args )

MODEL_NAME          = args.name
VALIDATION_SIZE     = args.validationsize
TOTAL_ITERATIONS    = args.niterations
VALIDATION_INTERVAL = args.validationinterval

# To make sure that we can reproduce the experiment and get the same results
np.random.seed(1234)

if platform.node() == 'chris-MS-7B09' :
    print("Using GPU options")
    tf_config = tf.ConfigProto( gpu_options = tf.GPUOptions(allow_growth=False),
                                allow_soft_placement = True )
else:
    tf_config = tf.ConfigProto()
#tf_config.log_device_placement=True
tf_config.intra_op_parallelism_threads = 16
tf_config.inter_op_parallelism_threads = 16
tf.reset_default_graph()

# create the model
rModel = RICH.createRICHModel()

# access data from the model
data_raw        = rModel["RawTrainData"]
data_train      = rModel["NormTrainData"]
val_raw         = rModel["RawValidationData"]
data_val        = rModel["NormValidationData"]
train_names     = rModel["InputNames"]
target_names    = rModel["TargetNames"]
dll_scaler      = rModel["DLLScaler"]
critic_train_op = rModel["CriticOptimizer"]
gen_train_op    = rModel["GeneratorOptimizer"]
tf_iter         = rModel["TfIterator"]
batch_gen_dlls  = rModel["BatchGeneratedDLLs"]
batch_data      = rModel["BatchInputs"]

# number outputs from generator
output_dim = len(target_names)

# Output directories
plots_dir = args.outputdir+"/"+MODEL_NAME+"/"
dirs = RICH.outputDirs( plots_dir )
print ( "Output dir", plots_dir )

# Make some input / output plots
RICH.initPlots( rModel, plots_dir )

merged_summary    = RICH.tfSummary( rModel )

validation_np     = data_val.sample(VALIDATION_SIZE)
validation_np_raw = val_raw.sample(VALIDATION_SIZE)

var_init      = tf.global_variables_initializer()
weights_saver = tf.train.Saver()
tf.get_default_graph().finalize()

MODEL_WEIGHTS_FILE = dirs["weights"]+"%s.ckpt" % MODEL_NAME
train_writer       = tf.summary.FileWriter(os.path.join(dirs["summary"], "train"))
test_writer        = tf.summary.FileWriter(os.path.join(dirs["summary"], "test"))

# functor to give the number of training runs per iteration
critic_policy = RICH.critic_policy(TOTAL_ITERATIONS)

with tf.Session(config=tf_config) as sess:

    # Initialise
    sess.run(var_init)
    
    # Try and restore a saved weights file
    #try:
    #    weights_saver.restore(sess, MODEL_WEIGHTS_FILE)
    #except tf.errors.NotFoundError:
    #    print("Can't restore parameters: no file with weights")

    # Do the iterations
    its = range(1,TOTAL_ITERATIONS+1)
    if not args.batchmode : its = tqdm(its)
    for i in its :

        for j in range(critic_policy(i)) :
            sess.run(critic_train_op)

        train_summary, _, interation = sess.run([merged_summary, gen_train_op, tf_iter])
        train_writer.add_summary(train_summary, interation)

        # Do validation now and then
        if i % VALIDATION_INTERVAL == 0 or i == 1 :

            # Directory for plots etc. for this iteratons
            it_dir = dirs["iterations"]+str( '%06d' % i )+"/"
            if not os.path.exists(it_dir) : os.makedirs(it_dir)

            #clear_output(False)
            test_summary, test_generated = sess.run( [merged_summary, batch_gen_dlls[0]], {
                batch_data[1]  : validation_np[train_names].values,
                batch_data[2]  : validation_np[train_names].values,
                batch_data[0]  : validation_np.values } )

            # Summary and weights
            test_writer.add_summary(test_summary, interation)
            weights_saver.save(sess, MODEL_WEIGHTS_FILE)

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


with tf.Session(config=tf_config) as sess:
    sess.run(var_init)
    weights_saver.restore(sess, MODEL_WEIGHTS_FILE)
    sess.graph._unsafe_unfinalize()
    tf.saved_model.simple_save(sess, dirs["model"],
                               inputs={"x": batch_data[1]}, outputs={"dlls": batch_gen_dlls[0]})
    tf.get_default_graph().finalize()
    
#from sklearn.externals import joblib
#joblib.dump(scaler, os.path.join(plots_dir, 'preprocessors', MODEL_NAME) + "_preprocessor.pkl")
