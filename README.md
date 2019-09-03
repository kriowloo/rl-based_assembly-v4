# rl-based_assembly-v4

**Introduction**

This version on RL-based assembly is using DQN to replace Q-learning in [Bocicor et al (2011)]( https://ieeexplore.ieee.org/document/6169520)<sup>1</sup> as an approach to DNA fragment assembly (DFA) problem. This DQN implementation is inspired on a published [manuscript of DeepMind](https://arxiv.org/abs/1312.5602)<sup>2</sup>, where DQN was firstly proposed by its researchers. It uses a convolutional neural network (CNN) with 2 convolutional layers followed by a fully connected layer which is connected to the output layer, that contains one output per each action of the RL model proposed in the aforementioned manuscript.

Once CNN requires images as inputs and this type of DQN requires a representation of an state as input, we transformed each state in the state space in a set of images containing one or two images. Bocicor et al proposed a state space where each state corresponds to a distinct arrange of reads with different amount of elements. It contains only one initial state, that represents the cenario where no read is present. As each read is incorporated, a new state emerge, composed by all reads used to reach the previous state and the last incorporated read (the action). 

This implementation is able to represent each state in different approaches. All of them first find the exact match between each pair of subsequent reads in the corresponding state. Using these matches, an image is created where its height corresponds to the number of reads to be assembled and width is the maximum width of an image where all reads overlap in only one character (preffix and suffix). Each nucleotide (A, C, T, and G) is also represented using specific pixel values. Five approaches are available, as briefly described below (a more comprehensive explanation is given here):

1) Two images for each state: Each state is represented by one image containing the alignments considering the forward order of the reads of this state and by other image for the reverse order. For example, if state X represents the arrangement of four reads A1-A5-A9-A4, on that order, it will produce 2 images with the same width and height: (a) one for the forward order A1-A5-A9-A4 and (b) another for the reverse order A4-A9-A5-A1.

2) Largest image for each state: The two images produced in the previous approach are analyzed and only the image which contains the largest alignment.

3) One merged image for each state: The two images produces in the first approach are merged in only one.

4) One random image for each state: One out of the two images produces in the first approach is randomly selected.

5) Only the forward image for each state: Only the forward image produced in the first approach is used to represent each state

> **References**
> 
> \[1\] M. Bocicor, G. Czibula and I. Czibula, "A Reinforcement Learning Approach for Solving the Fragment Assembly Problem," 2011 *13th International Symposium on Symbolic and Numeric Algorithms for Scientific Computing*, Timisoara, 2011, pp. 191-198. doi: 10.1109/SYNASC.2011.9
> 
> \[2\] V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller, “Playing Atari with Deep Reinforcement Learning,” *arXiv e-prints*, p. arXiv:1312.5602, Dec 2013.

**Install instructions**

1) Install prerequisites

   This software was written in Python and has the following dependencies:

   - Python 3 (or superior)
   - Git (optional)
   - Numpy (recommended version 1.14.3)
   - Biopython (recommended version 1.74)
   - Scipy (recommended version 1.1.0)
   - Pillow (recommended version 5.1.0)
   - TensorFlow (recommended version 1.8.0)
   - Keras (recommended version 2.1.6)
   
   
   > Note: We recommend to run this software into a container, since it requires specific versions of some Python modules. There is a Docker image already configured with Ubuntu (~1GB) able to run it. However, if you prefer, Python requirements file is also available into *src* folder and can be used to install all required modules through *pip*. To run it inside a container from the aforementioned Docker image, just run the following command:
   
      ```console
      user@host:~$ docker run --rm -it kpadovani/rlassembler-os:ubuntu
      ```

2) Clone this git repo (or manually download and extract files) into some folder (below, there is an example where git project is downloaded to the user's folder)

   ```console
   user@host:~# cd
   user@host:~# git clone https://github.com/kriowloo/rl-based_assembly-v4.git
   user@host:~# cd rl-based_assembly-v4
   ```
   
   > Note: After typing git clone, your username and password will be prompted, since this is a private project.

3) Run the software by using *RLAssembler.py* script (in *src* folder), setting up required parameters and providing the input file containing the reads to be assembled (in FASTA format). Run it without any parameters to see all parameter descriptions.

   ```console
   user@host:~# cd src
   user@host:~# python3 RLAssembler.py
   ```

   Below an example of execution using the example input file (named *experiment1.fasta* and stored in *data* folder), running for only 10 episodes on GPU and considering state representation number 5.
   
   ```console
   user@host:~# python3 RLAssembler.py -e 10 -bm 2000 -g 0.95 -em 0.01 -ed 0.995 -bb 32 -m 0 -s 5 -gpu 1 ../data/experiment1.fasta
   ```
**Documentation**

All source files are stored in *src* folder and were written in Python. There are 10 source files in this folder, as described below:

1) RLAssembly.py: This source file is responsible for starting the application. It checks input parameters from user and start RL training if everything is properly configured, by creating an object of Environment - that basically controls current state and allows action taking - and another object of DFADeepQNetwork - the class where all DQN strategy is implemented to DFA problem.

2)
