#! /usr/bin/env python3

import argparse, sys, os, platform
import time
import tensorflow as tf
import RICH

FLAGS = None

def main(_):

    def log(s) :
        if FLAGS.verbose : print(s)

    # Tf config
    tf_config = tf.ConfigProto()
    #tf_config.gpu_options = tf.GPUOptions(allow_growth=True)
    #tf_config.log_device_placement=True
    tf_config.intra_op_parallelism_threads = 16
    tf_config.inter_op_parallelism_threads = 16
    tf.reset_default_graph()

    ps_hosts     = FLAGS.ps_hosts.split(",") 
    worker_hosts = FLAGS.worker_hosts.split(",") 
    log( ps_hosts )
    log( worker_hosts )

    def create_done_queue(i):
        with tf.device("/job:ps/task:%d" % (i)) :
            return tf.FIFOQueue(len(worker_hosts), tf.int32, 
                                shared_name="done_queue"+str(i))
  
    def create_done_queues():
        return [ create_done_queue(i) for i in range(len(ps_hosts)) ]
    
    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec( { "ps": ps_hosts, "worker": worker_hosts } )
    
    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":

        #server.join()
        sess = tf.Session( server.target, config=tf_config )
        queue = create_done_queue(FLAGS.task_index)
        for i in range(len(worker_hosts)) : sess.run(queue.dequeue())

    elif FLAGS.job_name == "worker":

        # is this the master ?
        isChief = FLAGS.task_index == 0

        # If not the chief, sleep for 30 secs
        if not isChief : time.sleep(10)

        # Output directories
        plots_dir = FLAGS.outputdir+"/"+FLAGS.taskname+"/"
        #+platform.node()+"/"
        log( "Output dir " + plots_dir )
        dirs = RICH.outputDirs( plots_dir, isChief )

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            #global_step = tf.contrib.framework.get_or_create_global_step()
            global_step = tf.train.get_or_create_global_step()

            # create the model
            rModel = RICH.createRICHModel(global_step)

            # Make some input / output plots
            if isChief : RICH.initPlots( rModel, plots_dir )

            # input and output names
            train_names     = rModel["InputNames"]
            target_names    = rModel["TargetNames"]

            # data references
            data_raw        = rModel["RawTrainData"]
            data_train      = rModel["NormTrainData"]
            val_raw         = rModel["RawValidationData"]
            data_val        = rModel["NormValidationData"]
            batch_gen_dlls  = rModel["BatchGeneratedDLLs"]
            batch_data      = rModel["BatchInputs"]

            # data scalers
            dll_scaler      = rModel["DLLScaler"]

            # validation data
            validation_np     = data_val.sample(FLAGS.validationsize)
            validation_np_raw = val_raw.sample(FLAGS.validationsize)

            # optimizers
            critic_train_op = rModel["CriticOptimizer"]
            gen_train_op    = rModel["GeneratorOptimizer"]

            # Number of critic trainings per generator training
            critic_steps = 15

            # total steps
            total_steps = FLAGS.niterations

            # The StopAtStepHook handles stopping after running given steps.
            hooks = [ tf.train.StopAtStepHook( last_step = total_steps ) ]

            summary_op = RICH.tfSummary( rModel )

            enq_ops = []
            for q in create_done_queues():
                qop = q.enqueue(1)
                enq_ops.append(qop)

        lastMoniStep = 0
        step         = 0
                
        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(config=tf_config,
                                               master=server.target,
                                               is_chief=isChief,
                                               checkpoint_dir=dirs["checkpoint"],
                                               summary_dir=dirs["summary"],
                                               stop_grace_period_secs=120,
                                               max_wait_secs=7200,
                                               hooks=hooks) as mon_sess :

            while not mon_sess.should_stop() and step < total_steps :
      
                # Note. Need to double check 'should_stop()' before each run call.

                # Leave the last few steps to the chief otherwise it hangs...
                #if isChief or step < total_steps-10 :

                # Train the critic
                for j in range(critic_steps) : 
                    if not mon_sess.should_stop() :
                        mon_sess.run(critic_train_op)

                # Train the generator
                if not mon_sess.should_stop():
                    _, step = mon_sess.run( [gen_train_op,global_step] )
                else:
                    step = step + 1
                    
                log( "Global step %d" % step )
                
                # Do plots
                if isChief and step > 0 and (step-lastMoniStep) > FLAGS.validationinterval :

                    lastMoniStep = step
                    
                    log ( "Monitoring" )

                    # Directory for plots etc. for this iteratons
                    it_dir = dirs["iterations"]+str( '%06d' % step )+"/"
                    if not os.path.exists(it_dir) : os.makedirs(it_dir)

                    test_summary, test_generated = mon_sess.run( [summary_op, batch_gen_dlls[0]], {
                        batch_data[1]  : validation_np[train_names].values,
                        batch_data[2]  : validation_np[train_names].values,
                        batch_data[0]  : validation_np.values } )

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

        
            # Tell the ps servers
            #for q in create_done_queues() : sess.run(q.enqueue(1))
            #for op in enq_ops : mon_sess.run(op)
                   
                    
                    
     

if __name__ == "__main__":

    # print PID so handler script can capture it
    #print( os.getpid() )

    parser = argparse.ArgumentParser()

    parser.register("type", "bool", lambda v: v.lower() == "true")

    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )

    parser.add_argument( '--verbose', action='store_true' )
    parser.set_defaults(verbose=True)

    parser.add_argument( '--taskname', type=str, default="DistTest1" )
    parser.add_argument( '--outputdir', type=str, default="/usera/jonesc/NFS/output/MCGenGAN" )

    parser.add_argument( '--validationsize', type=int, default="100" )
    parser.add_argument( '--validationinterval', type=int, default="10" )
    parser.add_argument( '--niterations', type=int, default="20" )

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
