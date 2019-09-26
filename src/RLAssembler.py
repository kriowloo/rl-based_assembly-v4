
# requirements:
# => Python3
# => BioPython
# => Keras
# => Numpy

import sys
import os
from Bio import SeqIO

from Environment import Environment
from DFADeepQNetwork import DFADeepQNetwork
from State2TwoImages import State2TwoImages
from State2LargerImage import State2LargerImage
from State2MergedImage import State2MergedImage
from State2RandomImage import State2RandomImage
from State2ForwardImage import State2ForwardImage
from State2HiddenForwardImage import State2HiddenForwardImage

# start assembly after setting up params
def _start(param_values, reads, n_reads, max_read_len):
    print("Starting...")
    # print configuration of the experiment according to command-line params
    _showParams(param_values, n_reads, max_read_len)

    # retrieve command-line params
    frames_per_state = param_values["frames_per_state"]
    buffer_maxlen = param_values["buffer_maxlen"]
    epsilon = param_values["epsilon"]
    epsilon_decay = param_values["epsilon_decay"]
    epsilon_min = param_values["epsilon_min"]
    gamma = param_values["gamma"]
    threads = param_values["threads"]
    swmatch = param_values["swmatch"]
    swmismatch = param_values["swmismatch"]
    swgap = param_values["swgap"]
    stateversion = param_values["stateversion"]
    episodes = param_values["episodes"]
    buffer_batch_size = param_values["buffer_batch_size"]
    gpu_enabled = param_values["gpu_enabled"]
    max_actions_per_episode = param_values["max_actions_per_episode"]
    pixel_norm_type = param_values["pixel_norm_type"]

    # create a converter able to represent each state as image(s)
    print("Creating state2image converter...")
    ol = _getState2ImageConverter(stateversion, reads, max_read_len, swmatch, swmismatch, swgap, n_reads)

    # _, pm = ol._getCompressedImageForReads([9,8,7,6,5,4,3,2,1])
    # _, pm = ol._getCompressedImageForReads([9,0, 1, 3, 19, 20, 21, 27,8, 10, 23,29,22,25,11, 15, 17,2, 6, 7, 13, 14, 18,24, 26, 28,5,12, 16,4])
    # print(pm)
    # sys.exit(1)

    # create RL environment to assembly
    print("Creating assembly environment...")
    env = Environment(ol, reads, n_reads)

    # create intelligent agent
    print("Creating a DQN for DFA problem...")
    agent = DFADeepQNetwork(n_reads, max_read_len, frames_per_state, buffer_maxlen, epsilon, epsilon_decay, epsilon_min, gamma, threads, env, gpu_enabled, pixel_norm_type)

    # start training
    print("Starting training...")
    agent.train(episodes, buffer_batch_size, max_actions_per_episode)
    print("Training finished!")
    print("Actions after training:")
    agent.test(n_reads)
    print("Was that good?!")

# return an instance of a converter to transform any state in a set of images
def _getState2ImageConverter(sv, reads, max_read_len, match, mismatch, gap, n_reads):
    if sv == 1:
        return State2TwoImages(reads, max_read_len, match, mismatch, gap, n_reads)
    if sv == 2:
        return State2LargerImage(reads, max_read_len, match, mismatch, gap, n_reads)
    if sv == 3:
        return State2MergedImage(reads, max_read_len, match, mismatch, gap, n_reads)
    if sv == 4:
        return State2RandomImage(reads, max_read_len, match, mismatch, gap, n_reads)
    if sv == 5:
        return State2ForwardImage(reads, max_read_len, match, mismatch, gap, n_reads)
    if sv == 6:
        return State2HiddenForwardImage(reads, max_read_len, match, mismatch, gap, n_reads)
    return None

# show how to use the software
def printUsage(param_descriptions, required_params, param_tags, state_versions, pixel_norm_types, default_values):
    optional = {}
    print("Error to run RLAssembler.")
    print()
    print("Usage: python3 RLAssembler.py <param_name1> <param_value1> <param_name2> <param_value2> ... input.fasta")
    print("\tRequired params:")
    sorted_params = []
    for key in param_tags.keys():
        sorted_params.append(key)
    sorted_params.sort()
    for key in sorted_params:
        val = param_tags[key]
        cur_tag = key
        cur_val = param_descriptions[val]
        if val in default_values:
            cur_val += " [default: " + str(default_values[val]) + "]"
        if val in required_params:
            print("\t\t" + cur_tag + "\t" + cur_val)
        else:
            optional[cur_tag] = cur_val
    print("\tOptional params:")
    for key, val in optional.items():
        print("\t\t" + key + "\t" + val)
    print()
    print("State version corresponds to the representation of each state in state space. The following options are available:")
    sorted_state_versions = []
    for key in state_versions.keys():
        sorted_state_versions.append(key)
    sorted_state_versions.sort()
    for key in sorted_state_versions:
        val = state_versions[key]
        print("\t" + key + ": " + val + ";")

    print()
    print("Pixel normalization type corresponds to the mode each pixel will be represented to neural network. The following options are available:")
    sorted_pixel_norm_types = []
    for key in pixel_norm_types.keys():
        sorted_pixel_norm_types.append(key)
    sorted_pixel_norm_types.sort()
    for key in sorted_pixel_norm_types:
        val = pixel_norm_types[key]
        print("\t" + key + ": " + val + ";")
    
    print()

# setup parameters
# return True if all required params were correctly configured
def _setParams(param_tags, required_params, param_values, default_values):
    if len(sys.argv) <= 1 or len(sys.argv) % 2 != 0:
        return None, None, None
    required = set()
    for key, val in default_values.items():
        param_values[key] = val
    for i in range(1, len(sys.argv)-1, 2):
        tag = sys.argv[i]
        if tag not in param_tags:
            print("Warning: Param '" + tag + "' was not expected. Ignored!")
            continue
        param_name = param_tags[tag]
        value = _checkParam(param_name, sys.argv[i+1])
        if value is None:
            print("Warning: Value assigned to '" + param_name + "' is not valid. Ignored!")
            continue
        param_values[param_name] = value
        if param_name in required_params:
            required.add(param_name)
    if len(set(required_params)) != len(required):
        return None, None, None
    param_values["frames_per_state"] = 2 if param_values["stateversion"] == 1 else 1
    return _getReads(sys.argv[len(sys.argv) - 1])


# verify if a value is valid for a given tag (command-line parameter)
# return None if the value is incorrect
def _checkParam(tag, value):
    int_tags = ["episodes", "buffer_maxlen", "buffer_batch_size", "max_actions_per_episode", "threads", "gpu_enabled"]
    float_tags = ["swmatch", "swmismatch", "swgap"]
    perc_tags = ["gamma", "epsilon_min", "epsilon_decay", "epsilon"]
    if tag in int_tags:
        return int(value) if value.isdigit() else None
    if tag in float_tags:
        try:
            return float(value)
        except ValueError:
            return None
    if tag in perc_tags:
        try:
            v = float(value)
            return v if v >= 0 and v <= 1 else None
        except ValueError:
            return None
    if tag == "stateversion":
        if not value.isdigit():
            return None
        value = int(value)
        return value if value >= 1 and value <= 6 else None
    if tag == "pixel_norm_type":
        if not value.isdigit():
            return None
        value = int(value)
        return value if value >= 0 and value <= 1 else None

    return None

def _showParams(param_values, n_reads, max_read_len):
    print("Experimental setup:")
    for key, val in param_values.items():
        print("\t" + key + " = " + str(val))
    print("\tNumber of reads: " + str(n_reads))
    print("\tLargest read length: " + str(max_read_len))

def _getReads(fasta_path):
    print("Input file: " + fasta_path)
    if not os.path.isfile(fasta_path):
        print("Warning: Input FASTA not found.")
        return None, None, None
    try:
        max_read_len = 0
        reads = []
        with open(fasta_path, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                sequence = record.seq
                reads.append(sequence)
                if len(sequence) > max_read_len:
                    max_read_len = len(sequence)
        n_reads = len(reads)
        if n_reads == 0:
            print("Warning: No reads found in FASTA file.")
            return None, None, None
        return reads, n_reads, max_read_len
    except ValueError:
        return None, None, None

if __name__ == "__main__":
    param_descriptions = {
        "episodes" : "number of episodes for training",
        "buffer_maxlen" : "max number of last actions that will be stored (after reaching this maximum, older actions start to be discarded)",
        "gamma" : "q-learning discount rate (e.g.: 0.7)",
        "epsilon_min" : "minimum value epsilon can reach (e.g.: 0.01)",
        "epsilon_decay" : "a percentage to be preserved of the last epsilon value in each episode (e.g.: 0.995)",
        "buffer_batch_size" : "number of actions that will be sampled from the buffer after each episode",
        "max_actions_per_episode" : "limit the number of actions per each episode (set to zero to run with no limit)",
        "stateversion" : "set the type of state representation is going to be used (see valid values below)",
        "epsilon" : "e-greedy start value (exploration rate)",
        "swmatch" : "value for matches found in Smith-Waterman algorithm",
        "swmismatch" : "value for each occurrence of unmatch in Smith-Waterman algorithm",
        "swgap" : "value for gaps found in Smith-Waterman algorithm",
        "threads" : "number of threads to be used",
        "gpu_enabled" : "flag to indicate whether GPU is disabled (gpu_enabled=0) or enabled (gpu_enabled=positive)",
        "pixel_norm_type" : "set the type of pixel normalization is going to be used (see valid values below)"
    }
    # command-line parameters used to set up each parameter
    param_tags = {
        "-e" : "episodes", # number of episodes for training
        "-bm" : "buffer_maxlen", # max number of last actions that will be stored (after reaching this maximum, older actions start to be discarded)
        "-g" : "gamma", # q-learning discount rate (e.g.: 0.7)
        "-em" : "epsilon_min", # minimum value epsilon can reach (e.g.: 0.01)
        "-ed" : "epsilon_decay", # a percentage to be preserved of the last epsilon value in each episode (e.g.: 0.995)
        "-bb" : "buffer_batch_size", # number of actions that will be sampled from the buffer after each episode
        "-m" : "max_actions_per_episode", # limit the number of actions per each episode (set to zero to run with no limit)
        "-s" : "stateversion", # set the type of state representation is going to be used (2: two images for each state; 1L: one image for each state (that with largest width of type 1); 1M: one image for each state (both images from type 1 merged); 1R: one image for each state (one of the two images from type 1 is randomly choosen); 1S: one image for each state (only the image that read order correspond to the exact order found in the state))
        "-ep" : "epsilon", # e-greedy start value (exploration rate)
        "-swa" : "swmatch", # value for matches found in Smith-Waterman algorithm
        "-swi" : "swmismatch", # value for each occurrence of unmatch in Smith-Waterman algorithm
        "-swg" : "swgap", # value for gaps found in Smith-Waterman algorithm
        "-t" : "threads", # number of threads to be used
        "-gpu" : "gpu_enabled", # flag indicating the use (or not) of gpu
        "-norm" : "pixel_norm_type"
    }
    required_params = [
        "episodes",
        "buffer_maxlen",
        "gamma",
        "epsilon_min",
        "epsilon_decay",
        "buffer_batch_size",
        "max_actions_per_episode",
        "stateversion"
    ]
    # setting up default parameters
    default_values = {
        "epsilon" : 1.0,
        "swmatch" : 1.0,
        "swmismatch" : -0.33,
        "swgap" : -1.33,
        "threads" : 1,
        "gpu_enabled" : 0,
        "pixel_norm_type" : 0
    }
    # available options to represent each state
    state_versions = {
        "1": "two images for each state",
        "2": "one image for each state (that with largest width of type 1)",
        "3": "one image for each state (both images from type 1 merged)",
        "4": "one image for each state (one of the two images from type 1 is randomly choosen)",
        "5": "one image for each state (only the image that read order correspond to the exact order found in the state)",
        "6": "one image for each state (equals to version 5), but without misalignment between reads"        
    }
    pixel_norm_types = {
      "0": "convert pixel values by dividing actual values by 255 - thus black and white pixels will be normalized to zero and one respectively",
      "1": "convert pixel values so that white pixels will be normalized to zero and black pixels to one (formula: (255 - pixel) / 255)"
    }
    # dict to store all param values
    param_values = {}
    reads, n_reads, max_read_len = _setParams(param_tags, required_params, param_values, default_values)
    if reads is None:
        printUsage(param_descriptions, required_params, param_tags, state_versions, pixel_norm_types, default_values)
        sys.exit(1)

    _start(param_values, reads, n_reads, max_read_len)
