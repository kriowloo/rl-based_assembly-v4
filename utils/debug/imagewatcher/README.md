Set self.debug_episode = 0 at Environment class constructor and run your experiment. After the execution, PNG files will be stored into the execution folder. Move them to a new folder. Copy rescale_images.py and DejaVuSansMono-Bold.ttf files to this new folder and run rescale_images.py. To customize output, read comments inside this code.

After that, execute generate_video.sh. This execution will produce an output file named output.mp4 with the corresponding images. 

Note 1: Package ffmpeg has to be installed before executing this last step. If it is not installed and you are using Ubuntu, run apt-get install ffmpeg.

Note 2: If you chance default prefix to execute rescale_images.py, you will have to update generate_video.sh before running it.
