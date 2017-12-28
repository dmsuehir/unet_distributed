# These are the only things you need to change.
# Just replace the IP addresses with whatever machines you want to distribute over
# Then run this script on each of those machines.
"""
Usage:  python test_dist.py --ip=10.100.68.245 --is_sync=0
		for asynchronous TF
		python test_dist.py --ip=10.100.68.245 --is_sync=1
		for synchronous updates
		The IP address must match one of the ones in the list below. If not passed,
		then we"ll default to the current machine"s IP (which is usually correct unless you use OPA)
"""
import settings_dist

ps_hosts = settings_dist.PS_HOSTS
ps_ports = settings_dist.PS_PORTS
worker_hosts = settings_dist.WORKER_HOSTS
worker_ports = settings_dist.WORKER_PORTS

ps_list = ["{}:{}".format(x, y) for x, y in zip(ps_hosts, ps_ports)]
worker_list = [
    "{}:{}".format(x, y) for x, y in zip(worker_hosts, worker_ports)
]
print("Distributed TensorFlow training")
print("Parameter server nodes are: {}".format(ps_list))
print("Worker nodes are {}".format(worker_list))

CHECKPOINT_DIRECTORY = "./checkpoints"

####################################################################

import numpy as np
import tensorflow as tf
import os
import socket

# Fancy progress bar
from tqdm import tqdm
#tqdm.monitor_interval = 0
from tqdm import trange

from model import define_model, dice_coef_loss, dice_coef
from data import load_all_data, get_epoch
import multiprocessing

num_inter_op_threads = 2
num_intra_op_threads = settings_dist.NUM_INTRA_THREADS  #multiprocessing.cpu_count() // 2 # Use half the CPU cores

# Unset proxy env variable to avoid gRPC errors
del os.environ["http_proxy"]
del os.environ["https_proxy"]

# You can turn on the gRPC messages by setting the environment variables below
#os.environ["GRPC_VERBOSITY"]="DEBUG"
#os.environ["GRPC_TRACE"] = "all"

os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"
os.environ["OMP_NUM_THREADS"] = str(num_intra_op_threads)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate", settings_dist.LEARNINGRATE,
                          "Initial learning rate.")
tf.app.flags.DEFINE_integer("is_sync", 0, "Synchronous updates?")
tf.app.flags.DEFINE_string("ip", socket.gethostbyname(socket.gethostname()),
                           "IP address of this machine")
tf.app.flags.DEFINE_integer("batch_size", settings_dist.BATCH_SIZE,
                            "Batch size of input data")
tf.app.flags.DEFINE_integer("epochs", settings_dist.EPOCHS,
                            "Batch size of input data")
# Hyperparameters
learning_rate = FLAGS.learning_rate
batch_size = FLAGS.batch_size

if (FLAGS.ip in ps_hosts):
    job_name = "ps"
    task_index = ps_hosts.index(FLAGS.ip)
elif (FLAGS.ip in worker_hosts):
    job_name = "worker"
    task_index = worker_hosts.index(FLAGS.ip)
else:
    print(
        "Error: IP {} not found in the worker or ps node list.\nUse --ip= to specify which machine this is.".
        format(FLAGS.ip))
    exit()


def create_done_queue(i):
    """
    Queue used to signal termination of the i"th ps shard. 
    Each worker sets their queue value to 1 when done.
    The parameter server op just checks for this.
    """

    with tf.device("/job:ps/task:{}".format(i)):
        return tf.FIFOQueue(
            len(worker_hosts), tf.int32, shared_name="done_queue{}".format(i))


def create_done_queues():
    return [create_done_queue(i) for i in range(len(ps_hosts))]


def main(_):

    config = tf.ConfigProto(
        inter_op_parallelism_threads=num_inter_op_threads,
        intra_op_parallelism_threads=num_intra_op_threads)

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()  # For Tensorflow trace

    cluster = tf.train.ClusterSpec({"ps": ps_list, "worker": worker_list})
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    is_sync = (FLAGS.is_sync == 1)  # Synchronous or asynchronous updates
    is_chief = (task_index == 0)  # Am I the chief node (always task 0)

    greedy = tf.contrib.training.GreedyLoadBalancingStrategy(
        num_tasks=len(ps_hosts), load_fn=tf.contrib.training.byte_size_load_fn)

    if job_name == "ps":

        with tf.device(
                tf.train.replica_device_setter(
                    worker_device="/job:ps/task:{}".format(task_index),
                    ps_tasks=len(ps_hosts),
                    ps_strategy=greedy,
                    cluster=cluster)):

            sess = tf.Session(server.target, config=config)
            queue = create_done_queue(task_index)

            print("*" * 30)
            print("\nParameter server #{} on {}.\n\n" \
             "Waiting on workers to finish.\n\nPress CTRL-\\ to terminate early.\n"  \
             .format(task_index, ps_hosts[task_index]))
            print("*" * 30)

            # wait until all workers are done
            for i in range(len(worker_hosts)):
                sess.run(queue.dequeue())
                print("Worker #{} reports job finished.".format(i))

            print("Parameter server #{} is quitting".format(task_index))
            print("Training complete.")

    elif job_name == "worker":

        if is_chief:
            print("I am chief worker {} with task #{}".format(
                worker_hosts[task_index], task_index))
        else:
            print("I am worker {} with task #{}".format(
                worker_hosts[task_index], task_index))

        with tf.device(
                tf.train.replica_device_setter(
                    worker_device="/job:worker/task:{}".format(task_index),
                    ps_tasks=len(ps_hosts),
                    ps_strategy=greedy,
                    cluster=cluster)):
            global_step = tf.Variable(0, name="global_step", trainable=False)

            # Load the data
            imgs_train, msks_train, imgs_test, msks_test = load_all_data()
            train_length = imgs_train.shape[0]  # Number of train datasets
            test_length  = imgs_test.shape[0]   # Number of test datasets
            """
            BEGIN: Define our model
            """

            imgs = tf.placeholder(tf.float32, shape=(None,msks_train.shape[1],
                msks_train.shape[2],msks_train.shape[3]))

            msks = tf.placeholder(
                tf.float32,
                shape=(None, msks_train.shape[1], msks_train.shape[2],
                       msks_train.shape[3]))

            preds = define_model(
                imgs, False, settings_dist.OUT_CHANNEL_NO
            )  # Don't use upsampling. Instead use tranposed convolution.

            print('Model defined')
            

            loss_value = dice_coef_loss(msks, preds)
            dice_value = dice_coef(msks, preds)

            test_loss_value = dice_coef_loss(msks, preds)
            test_dice_value = dice_coef(msks, preds)

            """
            END: Define our model
            """

            # Define gradient descent optimizer
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate)

            grads_and_vars = optimizer.compute_gradients(loss_value)
            if is_sync:

                rep_op = tf.train.SyncReplicasOptimizer(
                    optimizer,
                    replicas_to_aggregate=len(worker_hosts),
                    total_num_replicas=len(worker_hosts),
                    use_locking=True)

                train_op = rep_op.apply_gradients(
                    grads_and_vars, global_step=global_step)

                init_token_op = rep_op.get_init_tokens_op()

                chief_queue_runner = rep_op.get_chief_queue_runner()

            else:

                train_op = optimizer.apply_gradients(
                    grads_and_vars, global_step=global_step)

            init_op = tf.global_variables_initializer()

            saver = tf.train.Saver()

            # These are the values we wish to print to TensorBoard
            
            tf.summary.scalar("loss", loss_value)
            tf.summary.histogram("loss", loss_value)
            tf.summary.scalar("dice", dice_value)
            tf.summary.histogram("dice", dice_value)

            tf.summary.image("predictions", preds, max_outputs=3)
            tf.summary.image("ground_truth", msks, max_outputs=3)
            tf.summary.image("images", imgs, max_outputs=3)

        # Need to remove the checkpoint directory before each new run
        # import shutil
        # shutil.rmtree(CHECKPOINT_DIRECTORY, ignore_errors=True)

        # Send a signal to the ps when done by simply updating a queue in the shared graph
        enq_ops = []
        for q in create_done_queues():
            qop = q.enqueue(1)
            enq_ops.append(qop)

        # Only the chief does the summary
        if is_chief:
            summary_op = tf.summary.merge_all()
        else:
            summary_op = None

        # Add summaries for test data
        test_loss_value = tf.placeholder(tf.float32, ())
        test_dice_value = tf.placeholder(tf.float32, ())

        test_loss_summary = tf.summary.scalar("loss_test", test_loss_value)
        test_dice_summary = tf.summary.scalar("dice_test", test_dice_value)

        # TODO:  Theoretically I can pass the summary_op into
        # the Supervisor and have it handle the TensorBoard
        # log entries. However, doing so seems to hang the code.
        # For now, I just handle the summary calls explicitly.
        import time
        sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir=CHECKPOINT_DIRECTORY + "/run" +
            time.strftime("_%Y%m%d_%H%M%S"),
            init_op=init_op,
            summary_op=None,
            saver=saver,
            global_step=global_step,
            save_model_secs=60
        )  # Save the model (with weights) everty 60 seconds

        # TODO:
        # I'd like to use managed_session for this as it is more abstract
        # and probably less sensitive to changes from the TF team. However,
        # I am finding that the chief worker hangs on exit if I use managed_session.
        with sv.prepare_or_wait_for_session(
                server.target, config=config) as sess:
            #with sv.managed_session(server.target) as sess:

            if is_chief and is_sync:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)

            print("Loading epoch")
            epoch = get_epoch(batch_size, imgs_train, msks_train)
            num_batches = len(epoch)
            print("Loaded")

            step = 0

            progressbar = trange(num_batches * FLAGS.epochs)
            last_step = 0

            while (not sv.should_stop()) and (
                    step < (num_batches * FLAGS.epochs)):

                batch_idx = step % num_batches # Which batch is the epoch?

                data = epoch[batch_idx, 0]
                labels = epoch[batch_idx, 1]

                # For n workers, break up the batch into n sections
                # Send each worker a different section of the batch
                data_range = int(batch_size / len(worker_hosts))
                start = data_range * task_index
                end = start + data_range

                feed_dict = {imgs: data[start:end], msks: labels[start:end]}

                history, loss_v, dice_v, step = sess.run(
                    [train_op, loss_value, dice_value, global_step],
                    feed_dict=feed_dict)

                # Print summary only on chief
                if (is_chief) :

                    summary = sess.run(summary_op, feed_dict=feed_dict)
                    sv.summary_computed(sess, summary)  # Update the summary

                    # Calculate metric on test dataset every epoch
                    if (batch_idx==0) and (step > num_batches):
                    
                        dice_v_test = 0.0
                        loss_v_test = 0.0

                        for idx in tqdm(range(0, imgs_test.shape[0] - batch_size, batch_size), 
                            desc="Calculating metrics on test dataset", leave=False):
                            x_test = imgs_test[idx:(idx+batch_size)] 
                            y_test = msks_test[idx:(idx+batch_size)] 

                            feed_dict = {imgs: x_test, msks: y_test}

                            l_v, d_v = sess.run([loss_value, dice_value], feed_dict=feed_dict)

                            dice_v_test += d_v / (test_length // batch_size)
                            loss_v_test += l_v / (test_length // batch_size)


                        print("\nEpoch {} of {}: Test loss = {:.4f}, Test Dice = {:.4f}" \
                            .format((step // num_batches), FLAGS.epochs,
                                loss_v_test, dice_v_test))

                        sv.summary_computed(sess, sess.run(test_loss_summary, 
                            feed_dict={test_loss_value:loss_v_test}) ) 
                        sv.summary_computed(sess, sess.run(test_dice_summary, 
                            feed_dict={test_dice_value:dice_v_test}) )  

                # Shuffle every epoch
                if (batch_idx==0) and (step > num_batches):

                    print("Shuffling epoch")
                    epoch = get_epoch(batch_size, imgs_train, msks_train)


                # Print the loss and dice metric in the progress bar.
                progressbar.set_description(
                    "(loss={:.4f}, dice={:.4f})".format(loss_v, dice_v))
                progressbar.update(step-last_step)
                last_step = step


            if (is_chief):

                dice_v_test = 0.0
                loss_v_test = 0.0

                for idx in tqdm(range(0, imgs_test.shape[0] - batch_size, batch_size), 
                    desc="Calculating final metrics on test dataset", leave=False):
                    x_test = imgs_test[idx:(idx+batch_size)] 
                    y_test = msks_test[idx:(idx+batch_size)] 

                    feed_dict = {imgs: x_test, msks: y_test}

                    l_v, d_v = sess.run([loss_value, dice_value], feed_dict=feed_dict)

                    dice_v_test += d_v / (test_length // batch_size)
                    loss_v_test += l_v / (test_length // batch_size)


                print("\nEpoch {} of {}: Test loss = {:.4f}, Test Dice = {:.4f}" \
                    .format((step // num_batches), FLAGS.epochs,
                        loss_v_test, dice_v_test))

                sv.summary_computed(sess, sess.run(test_loss_summary, 
                    feed_dict={test_loss_value:loss_v_test}) ) 
                sv.summary_computed(sess, sess.run(test_dice_summary, 
                    feed_dict={test_dice_value:dice_v_test}) )  

                # Save the meta graph
                meta_graph_def = tf.train.export_meta_graph(filename=CHECKPOINT_DIRECTORY+'/final_model.meta')

            # Send a signal to the ps when done by simply updating a queue in the shared graph
            for op in enq_ops:
                sess.run(
                    op
                )  # Send the "work completed" signal to the parameter server

        
        print("\n\nFinished work on this node.")

        sv.request_stop()
        #sv.stop()


if __name__ == "__main__":
    tf.app.run()



