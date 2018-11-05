#!/bin/bash

# environment
source /opt/rh/rh-python36/enable 
source /usera/jonesc/VirtualEnvs/TensorFlow/bin/activate

export OUTDIR="/usera/jonesc/NFS/output/MCGenGAN"
export JOBNAME="CondorTest1"
export LOGFILE=$OUTDIR"/"$JOBNAME"/out.log"

rm -rf $OUTDIR"/"$JOBNAME
mkdir -p $OUTDIR"/"$JOBNAME

# copy scripts into working dir
cp /usera/jonesc/Projects/MCGenGAN/LHCb/*.py .

# Job configuration

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
export MAXDATA=1000
export BATCH_SIZE=100
export VALIDATION_SIZE=100
export TOTAL_ITERATIONS=100
export VALIDATION_INTERVAL=10

# names of variables to extract for training data
#export INPUTS="NumLongTracks TrackP TrackPt"
#export INPUTS="NumRich1Hits NumRich2Hits TrackP TrackPt"
export INPUTS="NumRich1Hits NumRich2Hits TrackP TrackPt TrackRich1EntryX TrackRich1EntryY TrackRich1ExitX TrackRich1ExitY TrackRich2EntryX TrackRich2EntryY TrackRich2ExitX TrackRich2ExitY"
#train_names = [ 'NumRich1Hits', 'NumRich2Hits', 'TrackP', 'TrackPt' 
                #,'NumPVs'
                #,"NumLongTracks"
                #,"TrackChi2PerDof", "TrackNumDof"
                #,'TrackVertexX', 'TrackVertexY', 'TrackVertexZ' 
                #,'TrackRich1EntryX', 'TrackRich1EntryY' 
                #,'TrackRich1ExitX', 'TrackRich1ExitY'
                #,'TrackRich2EntryX', 'TrackRich2EntryY'
                #,'TrackRich2ExitX', 'TrackRich2ExitY' 

export OUTPUTS="RichDLLk"
#export OUTPUTS="RichDLLe RichDLLmu RichDLLk RichDLLp RichDLLd RichDLLbt"

# run the job
python run_tf.py --batchmode --name $JOBNAME --outputdir=$OUTDIR --datareadsize $MAXDATA --batchsize $BATCH_SIZE --validationsize $VALIDATION_SIZE --niterations $TOTAL_ITERATIONS --validationinterval $VALIDATION_INTERVAL --inputvars $INPUTS --outputvars $OUTPUTS 2>&1 | cat > $LOGFILE

exit 0
