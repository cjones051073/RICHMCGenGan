#! /usr/bin/env python3

import os, sys, paramiko, shutil, time

#cl023 CentOS Linux release 7.5.1804 (Core)
#cl024 CentOS Linux release 7.5.1804 (Core)
#cl025 CentOS Linux release 7.5.1804 (Core)
#cl026 CentOS Linux release 7.5.1804 (Core)
#pcjp CentOS Linux release 7.5.1804 (Core)
#pcjx CentOS Linux release 7.5.1804 (Core)
#pcka CentOS Linux release 7.5.1804 (Core)
#pcki CentOS Linux release 7.5.1804 (Core)
#pckw CentOS Linux release 7.5.1804 (Core)
#pclc CentOS Linux release 7.5.1804 (Core)
#pclq CentOS Linux release 7.5.1804 (Core)
#pcmf CentOS Linux release 7.5.1804 (Core)
#pcmg CentOS Linux release 7.5.1804 (Core)
#pcmm CentOS Linux release 7.5.1804 (Core)
#pcmq CentOS Linux release 7.5.1804 (Core)
#pcmx CentOS Linux release 7.5.1804 (Core)
#pcmy CentOS Linux release 7.5.1804 (Core)
#pcmz CentOS Linux release 7.5.1804 (Core)

port = "60000"

ps_hosts  = [ 'pcka', 'pcki' ]
wk_hosts  = [ 'pcmx', 'pcmy', 'pcmz',
              'pclc', 'pclq', 'pcmf',
              'pcmg', 'pcmm', 'pcmq',
              'pcjx' ]

#wk_hosts += [ 'cl026.hep.phy.private.cam.ac.uk' ]
#wk_hosts += [ 'cl023', 'cl024', 'cl025', 'cl026' ]

# Add ports 
ps_hosts_ports = [ h+":"+port for h in ps_hosts ]
wk_hosts_ports = [ h+":"+port for h in wk_hosts ]


INDIR   = "/usera/jonesc/Projects/MCGenGAN/"
OUTDIR  = "/usera/jonesc/NFS/output/MCGenGAN/"

JOBNAME = "DistTest1"

NITS      = 10000
DATAREAD  = 2000000
BATCHSIZE = 50000
VALSIZE   = 300000
VALINT    = 100

JOBDIR = OUTDIR + JOBNAME

LOGDIR  = JOBDIR + "/logs/"

#if     os.path.exists(JOBDIR) : shutil.rmtree(JOBDIR) 
if not os.path.exists(JOBDIR) : os.makedirs(JOBDIR)
if not os.path.exists(LOGDIR) : os.makedirs(LOGDIR)

CommmonOpts  = "--datareadsize="+str(DATAREAD)+" --batchsize="+str(BATCHSIZE)+" --niterations="+str(NITS)
CommmonOpts += " --validationsize="+str(VALSIZE)+" --validationinterval="+str(VALINT)

CommmonOpts += " --ps_hosts="+",".join(ps_hosts_ports)
CommmonOpts += " --worker_hosts="+",".join(wk_hosts_ports)

def ssh(host) :
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    #print ( "Connecting to", host )
    client.connect( host )
    return client

ssh_threads = [ ]

# Start ps processes
ps_id = 0
for h in ps_hosts :

    print( "Starting ps task", ps_id, "on", h )
    
    # Open ssh session to host and save connection
    client = ssh(h)
    ssh_threads += [ client ]

    # log file for this process
    LOGFILE = LOGDIR + "/" + h + ".log"

    # construct the command and run it
    cmd  = 'source '+INDIR+'setup.sh ; nice -10 '+INDIR+'LHCb/run_tf_cluster.py --job_name=ps '+CommmonOpts
    cmd += ' --task_index='+str(ps_id)
    cmd += ' 2>&1 | cat > '+LOGFILE
    cmd += ' &'

    #print( cmd )
    client.exec_command(cmd)
 
    # increment ps ID for next process
    ps_id = ps_id + 1
    

# Start wk processes
ps_id = 0
for h in wk_hosts :

    print( "Starting worker task", ps_id, "on", h )

    # Open ssh session to host and save connection
    client = ssh(h)
    ssh_threads += [ client ]

    # log file for this process
    LOGFILE = LOGDIR + "/" + h + ".log"

    # construct the command and run it
    cmd  = 'source '+INDIR+'setup.sh ; nice -10 '+INDIR+'LHCb/run_tf_cluster.py --job_name=worker '+CommmonOpts
    cmd += ' --task_index='+str(ps_id)
    cmd += ' 2>&1 | cat > '+LOGFILE
    cmd += ' &'

    #print( cmd )
    client.exec_command(cmd)
 
    # increment ps ID for next process
    ps_id = ps_id + 1


# wait forever...
print( "Will run until you contrl-c me..." )
while True : 

    try:

        time.sleep(60)

    except KeyboardInterrupt:

        print("Caught control-C")

        for ssh in ssh_threads :

            # send kill command to this machine
            cmd = "kill `ps -fu jonesc | grep run_tf_cluster | grep -v grep  | awk '{print $2}'`"
            ssh.exec_command(cmd)

            ssh.close()

        exit(0)
