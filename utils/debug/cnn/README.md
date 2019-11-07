These codes were developed to analyze the learning process of the proposed CNN to model DQN algorithm for the assembly problem. They were created to evaluate whether the network is able to learn under a controlled environment. So that, input images and corresponding labels (expected values) were first uniformly provided to the CNN, without noises. Then, the order of images was changed randomly. Experiments showed us that the CNN was able to associate input images with their corresponding output.

Files are organized as follows:

- grayscale_images: Experiments with input images drawed in gray scale tones (each nucleotide has a specific gray tone)
-- default_reward: Experiments using the previous reward established (partial and integral PM)
--- testWhite.py: Script that train CNN considering that zero (in input image) corresponds to white color
--- testBlack.py: Script that train CNN considering that zero (in input image) corresponds to black color
--- XX%number.png: Input images used to train CNN (`number` corresponds to the reward obtained when the state represented by this image is reached and `XX` corresponds to the order of optimal state)
-- new_reward: Experiments using a reward based on suffix-prefix score normalized (same structure as `default_reward`)
- binary_images: Experiments with input images drawed in black and white (all nucleotides are represented as black) (same structure as `grayscale_images`)
- Folder old_versions: previous versions of the final codes (*Buffered.py corresponds to the first update to implement multiframes and other files correspond to the first version, without Double DQN and multiframes)

