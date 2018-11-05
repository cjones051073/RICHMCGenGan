#!/bin/bash

# environment
source /opt/rh/rh-python36/enable 
source /usera/jonesc/VirtualEnvs/TensorFlow/bin/activate

export OUTDIR="/usera/jonesc/NFS/output/MCGenGAN"

export JOBNAME="CondorTest1"

export JOBDIR=$OUTDIR"/"$JOBNAME
mkdir -p $JOBDIR
cd $JOBDIR

export LOGFILE=$JOBDIR"/out.log"

rm -rf $JOBDIR
mkdir -p $JOBDIR

# copy scripts into working dir
cp ${HOME}/Projects/MCGenGAN/LHCb/{RICH,run_tf}.py .

# Job configuration

# Standard large job
#maxData                 = -1
#BATCH_SIZE              = int(5e4)
#VALIDATION_SIZE         = int(3e5)
#TOTAL_ITERATIONS        = int(1e4)
#VALIDATION_INTERVAL     = 100

# tiny test job
#export MAXDATA=1000
#export BATCH_SIZE=100
#export VALIDATION_SIZE=100
#export TOTAL_ITERATIONS=100
#export VALIDATION_INTERVAL=10

# medium job
export MAXDATA=-1
export BATCH_SIZE=1000
export VALIDATION_SIZE=100
export TOTAL_ITERATIONS=1000
export VALIDATION_INTERVAL=100

# long job
#export MAXDATA=-1
#export BATCH_SIZE=50000
#export VALIDATION_SIZE=300000
#export TOTAL_ITERATIONS=10000
#export VALIDATION_INTERVAL=100

# names of variables to extract for training data
export INPUTS="NumLongTracks TrackP TrackPt"
#export INPUTS="NumRich1Hits NumRich2Hits TrackP TrackPt"
#export INPUTS="NumRich1Hits NumRich2Hits TrackP TrackPt TrackRich1EntryX TrackRich1EntryY TrackRich1ExitX TrackRich1ExitY TrackRich2EntryX TrackRich2EntryY TrackRich2ExitX TrackRich2ExitY"
#train_names = [ 'NumRich1Hits', 'NumRich2Hits', 'TrackP', 'TrackPt' 
                #,'NumPVs'
                #,"NumLongTracks"
                #,"TrackChi2PerDof", "TrackNumDof"
                #,'TrackVertexX', 'TrackVertexY', 'TrackVertexZ' 
                #,'TrackRich1EntryX', 'TrackRich1EntryY' 
                #,'TrackRich1ExitX', 'TrackRich1ExitY'
                #,'TrackRich2EntryX', 'TrackRich2EntryY'
                #,'TrackRich2ExitX', 'TrackRich2ExitY' 

#export OUTPUTS="RichDLLk"
#export OUTPUTS="RichDLLe RichDLLmu RichDLLk RichDLLp RichDLLd RichDLLbt"
export OUTPUTS="RichDLLe RichDLLk RichDLLp RichDLLbt"

# run the job
./run_tf.py --batchmode --name $JOBNAME --outputdir=$OUTDIR --datareadsize $MAXDATA --batchsize $BATCH_SIZE --validationsize $VALIDATION_SIZE --niterations $TOTAL_ITERATIONS --validationinterval $VALIDATION_INTERVAL --inputvars $INPUTS --outputvars $OUTPUTS 2>&1 | cat > $LOGFILE

exit 0
