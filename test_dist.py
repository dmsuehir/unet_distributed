#!/usr/bin/env python
# These are the only things you need to change.
# Just replace the IP addresses with whatever machines you want to
# distribute over
# Then run this script on each of those machines.

import json
import logging
import multiprocessing
import numpy as np
import os
import settings_dist
import signal
import socket
import subprocess
import tensorflow as tf
import time

# Fancy progress bar
from tqdm import tqdm
# tqdm.monitor_interval = 0
from tqdm import trange

from model import define_model, dice_coef_loss, dice_coef,\
    sensitivity, specificity
from data import load_all_data, get_epoch


CHECKPOINT_DIRECTORY = settings_dist.CHECKPOINT_DIRECTORY

num_inter_op_threads = settings_dist.NUM_INTER_THREADS
num_intra_op_threads = settings_dist.NUM_INTRA_THREADS

# multiprocessing.cpu_count() // 2 # Use half the CPU cores

# Unset proxy env variable to avoid gRPC errors
try:
    del os.environ["http_proxy"]
    del os.environ["https_proxy"]
except Exception:
    pass

# You can turn on the gRPC messages by setting the environment variables below
# os.environ["GRPC_VERBOSITY"]="DEBUG"
# os.environ["GRPC_TRACE"] = "all"

os.environ["KMP_BLOCKTIME"] = str(settings_dist.BLOCKTIME)
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"
os.environ["OMP_NUM_THREADS"] = str(num_intra_op_threads)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean("const_learningrate",
                            settings_dist.CONST_LEARNINGRATE,
                            "Keep learning rate constant or "
                            "exponentially decay")
tf.app.flags.DEFINE_float("learning_rate", settings_dist.LEARNINGRATE,
                          "Initial learning rate.")

tf.app.flags.DEFINE_float("lr_fraction", settings_dist.LR_FRACTION,
                          "Learning rate fraction for decay")
tf.app.flags.DEFINE_integer("decay_steps", settings_dist.DECAY_STEPS,
                            "Number of steps for decay")


tf.app.flags.DEFINE_integer("is_sync", 1, "Synchronous updates?")
tf.app.flags.DEFINE_string("ip", socket.gethostbyname(socket.gethostname()),
                           "IP address of this machine")
tf.app.flags.DEFINE_integer("batch_size", settings_dist.BATCH_SIZE,
                            "Batch size of input data")
tf.app.flags.DEFINE_integer("epochs", settings_dist.EPOCHS,
                            "Number of epochs to train")

tf.app.flags.DEFINE_boolean("use_upsampling", settings_dist.USE_UPSAMPLING,
                            "True = Use upsampling; False = Use transposed "
                            "convolution")

# Hyperparameters
batch_size = FLAGS.batch_size

time_left_to_train = 0  # Number of seconds left in training

# Get Cluster info from the TF_CONFIG environment variable
tf_config_json = os.environ.get("TF_CONFIG", "{}")
tf_config = json.loads(tf_config_json)
print("tf_config: {}".format(tf_config))

task = tf_config.get("task", "{}")
task_index = task.get("index", "{}")
job_name = task.get("type", "{}")
cluster = tf_config.get("cluster", "{}")

print("cluster: {}".format(cluster))

worker_list = worker_hosts = cluster.get("worker", "{}")
ps_list = ps_hosts = cluster.get("ps", "{}")

print("Parameter server nodes are: {}".format(ps_list))
print("Worker nodes are {}".format(worker_list))
print("Job name: {}".format(job_name))
print("Task index: {}".format(task_index))


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
    tf.logging.info("Distributed TensorFlow training")

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
        tf.logging.info("I'm the parameter server")

        with tf.device(
                tf.train.replica_device_setter(
                    worker_device="/job:ps/task:{}".format(task_index),
                    ps_tasks=len(ps_hosts),
                    ps_strategy=greedy,
                    cluster=cluster)):

            tf.logging.info("tf.Session")
            sess = tf.Session(server.target, config=config)
            tf.logging.info("create done queue")
            queue = create_done_queue(task_index)

            tf.logging.info("*" * 30)
            tf.logging.info("\nParameter server #{} on {}.\n\nWaiting on "
                            "workers to finish.\n\nPress CTRL-\\ to terminate "
                            "early.\n".format(task_index,
                                              ps_hosts[task_index]))
            tf.logging.info("*" * 30)

            # wait until all workers are done
            for i in range(len(worker_hosts)):
                sess.run(queue.dequeue())
                tf.logging.info("Worker #{} reports job finished.".format(i))

            tf.logging.info("Parameter server #{} is quitting".
                            format(task_index))
            tf.logging.info("Training complete.")

    elif job_name == "worker":

        if is_chief:
            tf.logging.info("I am chief worker {} with task #{}".format(
                worker_hosts[task_index], task_index))
        else:
            tf.logging.info("I am worker {} with task #{}".format(
                worker_hosts[task_index], task_index))

        if len(ps_list) > 0:
            setDevice = tf.train.replica_device_setter(
                     worker_device="/job:worker/task:{}".format(task_index),
                     ps_tasks=len(ps_hosts),
                     ps_strategy=greedy,
                     cluster=cluster)
        else:
            # No parameter server so put variables on chief worker
            setDevice = "/cpu:0"

        with tf.device(setDevice):

            global_step = tf.Variable(0, name="global_step", trainable=False)

            # Load the data
            imgs_train, msks_train, imgs_test, msks_test = load_all_data()
            train_length = imgs_train.shape[0]  # Number of train datasets
            test_length = imgs_test.shape[0]   # Number of test datasets

            """
            BEGIN: Define our model
            """

            imgs = tf.placeholder(tf.float32, shape=(None, msks_train.shape[1],
                                                     msks_train.shape[2],
                                                     msks_train.shape[3]))

            msks = tf.placeholder(
                tf.float32,
                shape=(None, msks_train.shape[1], msks_train.shape[2],
                       msks_train.shape[3]))

            preds = define_model(
                imgs, FLAGS.use_upsampling, settings_dist.OUT_CHANNEL_NO
            )

            tf.logging.info('Model defined')

            loss_value = dice_coef_loss(msks, preds)
            dice_value = dice_coef(msks, preds)

            sensitivity_value = sensitivity(msks, preds)
            specificity_value = specificity(msks, preds)

            test_loss_value = tf.placeholder(tf.float32, ())
            test_dice_value = tf.placeholder(tf.float32, ())

            test_sensitivity_value = tf.placeholder(tf.float32, ())
            test_specificity_value = tf.placeholder(tf.float32, ())

            """
            END: Define our model
            """

            # Decay learning rate from initial_learn_rate to
            # initial_learn_rate*fraction in decay_steps global steps
            if FLAGS.const_learningrate:
                learning_rate = tf.convert_to_tensor(
                    FLAGS.learning_rate, dtype=tf.float32)
            else:
                learning_rate = tf.train.exponential_decay(
                    FLAGS.learning_rate, global_step, FLAGS.decay_steps,
                    FLAGS.lr_fraction, staircase=False)

            # Compensate learning rate for asynchronous distributed
            # THEORY: We need to cut the learning rate by at least the number
            # of workers since there are likely to be that many times increased
            # parameter updates.
            # if not is_sync:
            #     learning_rate /= len(worker_hosts)
            #     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            #     #optimizer = tf.train.AdagradOptimizer(learning_rate)
            # else:
            #     optimizer = tf.train.AdamOptimizer(learning_rate)

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

            tf.summary.scalar("sensitivity", sensitivity_value)
            tf.summary.histogram("sensitivity", sensitivity_value)
            tf.summary.scalar("specificity", specificity_value)
            tf.summary.histogram("specificity", specificity_value)

            tf.summary.image("predictions", preds,
                             max_outputs=settings_dist.TENSORBOARD_IMAGES)
            tf.summary.image("ground_truth", msks,
                             max_outputs=settings_dist.TENSORBOARD_IMAGES)
            tf.summary.image("images", imgs,
                             max_outputs=settings_dist.TENSORBOARD_IMAGES)

            tf.logging.info("Loading epoch")
            epoch = get_epoch(batch_size, imgs_train, msks_train)
            num_batches = len(epoch)
            tf.logging.info("Loaded")

            # Print the percent steps complete to TensorBoard
            #   so that we know how much of the training remains.
            num_steps_tf = tf.constant(num_batches * FLAGS.epochs, tf.float32)
            percent_done_value = tf.constant(100.0) * tf.to_float(global_step)\
                / num_steps_tf
            tf.summary.scalar("percent_complete", percent_done_value)

        # Need to remove the checkpoint directory before each new run
        # import shutil
        # shutil.rmtree(CHECKPOINT_DIRECTORY, ignore_errors=True)

        # Send a signal to the ps when done by simply updating a queue
        # in the shared graph
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
        # These summary ops are not part of the merge all op.
        # This way we can call these separately.
        test_loss_value = tf.placeholder(tf.float32, ())
        test_dice_value = tf.placeholder(tf.float32, ())

        test_loss_summary = tf.summary.scalar("loss_test", test_loss_value)
        test_dice_summary = tf.summary.scalar("dice_test", test_dice_value)

        test_sens_summary = tf.summary.scalar("sensitivity_test",
                                              test_sensitivity_value)
        test_spec_summary = tf.summary.scalar("specificity_test",
                                              test_specificity_value)

        # TODO:  Theoretically I can pass the summary_op into
        # the Supervisor and have it handle the TensorBoard
        # log entries. However, doing so seems to hang the code.
        # For now, I just handle the summary calls explicitly.
        # import time
        # logDirName = CHECKPOINT_DIRECTORY + "/run" + \
        #             time.strftime("_%Y%m%d_%H%M%S")

        if FLAGS.use_upsampling:
            method_up = "upsample2D"
        else:
            method_up = "conv2DTranspose"

        logDirName = CHECKPOINT_DIRECTORY + "/unet," \
                                            "lr={},{},intra={},inter={}".\
            format(FLAGS.learning_rate, method_up, num_intra_op_threads,
                   num_inter_op_threads)

        sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir=logDirName,
            init_op=init_op,
            summary_op=None,
            saver=saver,
            global_step=global_step,
            save_model_secs=60  # Save the model (with weights) every 60 sec
        )

        # TODO:
        # I'd like to use managed_session for this as it is more abstract
        # and probably less sensitive to changes from the TF team. However,
        # I am finding that the chief worker hangs on exit if I use
        # managed_session.
        with sv.prepare_or_wait_for_session(
                server.target, config=config) as sess:
            # with sv.managed_session(server.target) as sess:

            if sv.is_chief and is_sync:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)

            step = 0

            progressbar = trange(num_batches * FLAGS.epochs)
            last_step = 0

            # Start TensorBoard on the chief worker
            if sv.is_chief:
                cmd = 'tensorboard --logdir={}'.format(CHECKPOINT_DIRECTORY)
                # tf.logging.info("Start TensorBoard by running: "
                #                 "{}".format(cmd))
                tb_process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                              shell=True,
                                              preexec_fn=os.setsid)

            while (not sv.should_stop()) and (
                    step < (num_batches * FLAGS.epochs)):

                batch_idx = step % num_batches  # Which batch is the epoch?

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
                if sv.is_chief:

                    summary = sess.run(summary_op, feed_dict=feed_dict)
                    sv.summary_computed(sess, summary)  # Update the summary

                    # Calculate metric on test dataset every epoch
                    if (batch_idx == 0) and (step > num_batches):

                        dice_v_test = 0.0
                        loss_v_test = 0.0
                        sens_v_test = 0.0
                        spec_v_test = 0.0

                        for idx in tqdm(range(0,
                                              imgs_test.shape[0] - batch_size,
                                              batch_size),
                                        desc="Calculating metrics on test "
                                             "dataset",
                                        leave=False):
                            x_test = imgs_test[idx:(idx+batch_size)]
                            y_test = msks_test[idx:(idx+batch_size)]

                            feed_dict = {imgs: x_test, msks: y_test}

                            l_v, d_v, st_v, sp_v = sess.run(
                                [loss_value, dice_value, sensitivity_value,
                                 specificity_value],
                                feed_dict=feed_dict)

                            dice_v_test += d_v / (test_length // batch_size)
                            loss_v_test += l_v / (test_length // batch_size)
                            sens_v_test += st_v / (test_length // batch_size)
                            spec_v_test += sp_v / (test_length // batch_size)

                        tf.logging.info("\nEpoch {} of {}: TEST DATASET\nloss"
                                        " = {:.4f}\nDice = {:.4f}\nSensitivity"
                                        " = {:.4f}\nSpecificity = {:.4f}".
                                        format((step // num_batches),
                                               FLAGS.epochs, loss_v_test,
                                               dice_v_test, sens_v_test,
                                               spec_v_test))

                        # Add our test summary metrics to TensorBoard
                        sv.summary_computed(sess, sess.run(
                            test_loss_summary,
                            feed_dict={test_loss_value: loss_v_test}))
                        sv.summary_computed(sess, sess.run(
                            test_dice_summary,
                            feed_dict={test_dice_value: dice_v_test}))
                        sv.summary_computed(sess, sess.run(
                            test_sens_summary,
                            feed_dict={test_sensitivity_value: sens_v_test}))
                        sv.summary_computed(sess, sess.run(
                            test_spec_summary,
                            feed_dict={test_specificity_value: spec_v_test}))

                        saver.save(sess,
                                   CHECKPOINT_DIRECTORY +
                                   "/last_good_model.cpkt")

                # Shuffle every epoch
                if (batch_idx == 0) and (step > num_batches):

                    tf.logging.info("Shuffling epoch")
                    epoch = get_epoch(batch_size, imgs_train, msks_train)

                # Print the loss and dice metric in the progress bar.
                progressbar.set_description(
                    "(loss={:.4f}, dice={:.4f})".format(loss_v, dice_v))
                progressbar.update(step-last_step)
                last_step = step

            # Perform the final test set metric
            if sv.is_chief:

                dice_v_test = 0.0
                loss_v_test = 0.0

                for idx in tqdm(range(0, imgs_test.shape[0] - batch_size,
                                      batch_size),
                                desc="Calculating metrics on test dataset",
                                leave=False):
                    x_test = imgs_test[idx:(idx+batch_size)]
                    y_test = msks_test[idx:(idx+batch_size)]

                    feed_dict = {imgs: x_test, msks: y_test}

                    l_v, d_v = sess.run([loss_value, dice_value],
                                        feed_dict=feed_dict)

                    dice_v_test += d_v / (test_length // batch_size)
                    loss_v_test += l_v / (test_length // batch_size)

                tf.logging.info("\nEpoch {} of {}: Test loss = {:.4f}, "
                                "Test Dice = {:.4f}".
                                format((step // num_batches),
                                       FLAGS.epochs, loss_v_test, dice_v_test))

                sv.summary_computed(sess, sess.run(
                    test_loss_summary,
                    feed_dict={test_loss_value: loss_v_test}))
                sv.summary_computed(sess, sess.run(
                    test_dice_summary,
                    feed_dict={test_dice_value: dice_v_test}))

                saver.save(sess,
                           CHECKPOINT_DIRECTORY + "/last_good_model.cpkt")

            if sv.is_chief:
                # Save the final model as protbuf for TensorFlow Serving
                export_model(sess, imgs, preds)

                # Stop TensorBoard process
                try:
                    os.killpg(os.getpgid(tb_process.pid), signal.SIGTERM)
                except NameError:
                    pass  # tf_process probably was not started

            # Send a signal to the ps when done by simply updating a queue
            # in the shared graph
            for op in enq_ops:
                sess.run(
                    op
                )  # Send the "work completed" signal to the parameter server

        tf.logging.info("\n\nFinished work on this node.")
        time.sleep(3)  # Sleep for 3 seconds then exit

        sv.request_stop()
        sv.stop()


def export_model(sess, input_tensor, output_tensor):

    # To view pb model file:  saved_model_cli show --dir saved_model --all
    sess.graph._unsafe_unfinalize()
    import shutil
    MODEL_DIR = CHECKPOINT_DIRECTORY + "saved_model"
    shutil.rmtree(MODEL_DIR, ignore_errors=True)  # Remove old saved model
    builder = tf.saved_model.builder.SavedModelBuilder(MODEL_DIR)

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "intel_unet_brats_model":
                tf.saved_model.signature_def_utils.predict_signature_def(
                    inputs={"image": input_tensor},
                    outputs={"prediction": output_tensor})},
        clear_devices=True)

    builder.save()

    tf.logging.info("Saved final model to directory: {}".format(MODEL_DIR))
    tf.logging.info("You can check the model from the command line "
                    "by running:")
    tf.logging.info("saved_model_cli show --dir {} --all".format(MODEL_DIR))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
