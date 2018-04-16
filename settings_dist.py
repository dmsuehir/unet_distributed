import os

BASE = "/home/bduser/unet/data/"
# DATA_PATH = BASE+"/slices"
DATA_PATH = "/tmp/data"
# OUT_PATH  = BASE+"slices/Results/"
OUT_PATH = "/tmp/data"
IMG_ROWS = 128
IMG_COLS = 128
RESCALE_FACTOR = 1
SLICE_BY = 5

IN_CHANNEL_NO = 1
OUT_CHANNEL_NO = 1

EPOCHS = 10

BLOCKTIME = 0
NUM_INTRA_THREADS = 50
NUM_INTER_THREADS = 2
BATCH_SIZE = 128

LEARNINGRATE = 0.0005  # 0.0005
DECAY_STEPS = 100
LR_FRACTION = 0.2
CONST_LEARNINGRATE = True

USE_UPSAMPLING = False
# True = Use upsampling; False = Use transposed convolution

MODEL_FN = "brainWholeTumor"  # Name for Mode=1
# MODEL_FN = "brainActiveTumor" #Name for Mode=2
# MODEL_FN = "brainCoreTumor" #Name for Mode=3

# Use flair to identify the entire tumor: test reaches 0.78-0.80: MODE=1
# Use T1 Post to identify the active tumor: test reaches 0.65-0.75: MODE=2
# Use T2 to identify the active core (necrosis, enhancing, non-enh):
# test reaches 0.5-0.55: MODE=3
MODE = 1

# Important that these are ordered correctly:
# [0] = chief node, [1] = worker node, etc.
# PS_HOSTS = ["10.100.68.245"]
# PS_PORTS = ["2222"]
# WORKER_HOSTS =
# ["10.100.68.193","10.100.68.183","10.100.68.185","10.100.68.187"]
# WORKER_PORTS = ["2222", "2222", "2222", "2222"]

# CHECKPOINT_DIRECTORY = "s3://dmsuehir/data03/checkpoints/"
CHECKPOINT_DIRECTORY = os.environ.get("OUTPUT_PATH", "/data03/checkpoints/")
TENSORBOARD_IMAGES = 3  # How many images to display on TensorBoard

# TensorBoard
# To run TensorBoard you must log into the chief worker (first one in the
# WORKER_HOSTS list).  Start your Tensorflow virtual environment and run
# `tensorboard --logdir=checkpoints` where checkpoints is whatever
# directory holds your log files. On your local machine (the one where you
# can run Chrome web browser), run the command: `ssh -f bduser@10.54.68.193
# -L 6006:localhost:6006 -N` where the `bduser@10.54.68.193` is replaced
# with the username and IP of the chief worker. Then on the local machine
# start Chrome webbrowser and go to url  http://localhost:6006

# RegEx to exclude plots that match "test", "step", and "complete":
#  ^((?!test)(?!step)(?!complete).)*$
