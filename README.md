# rl-based_assembly-v4

This version on RL-based assembly is using DQN to replace Q-learning in Bocicor et al (2011) as an approach to DNA fragment assembly (DFA) problem. This implementation is inspired on https://arxiv.org/abs/1312.5602, where DQN was first proposed by DeepMind researchers. It uses a convolutional neural network (CNN) with 2 convolutional layers followed by a fully connected layer which is connected to the output layer, that contains one output per each action of the RL model proposed in the aforementioned manuscript.

Once CNN requires images as inputs and this type of DQN requires a representation of an state as input, we transformed each state in the state space in a set of images containing one or two images. Bocicor et al proposed a state space where each state corresponds to a distinct arrange of reads with different amount of elements. It contains only one initial state, that represents the cenario where no read is present. As each read is incorporated, a new state emerge, composed by all reads used to reach the previous state and the last incorporated read (the action). 

This implementation is able to represent each state in 5 different approaches, as described below:

1) Two images for each state: in this approach, 
