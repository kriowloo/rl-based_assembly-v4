To produce videos, print the output of testWhite.py/testBlack.py to a text file named `saida.txt` (eg.: python3 testZeroWhiteBufferedShuffled.py 1200 1 3 200 > saida.txt). This file has to be saved into this folder and will be used by modify.py in the future.

Before running modify.py, remove all png files inside this folder and run the following:

1) If you executed the experiment on gray scale images:
cp background.png.gray background.png

2) If you executed the experiment on binary images:
cp background.png.black background.png

Now, run python3 modify.py (it is not fast).

After running it, execute the following command (it requires ffmpeg to be installed):

ffmpeg -pattern_type glob -framerate 30 -i "*.png" -c:v libx264 -profile:v high -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y output.mp4 

It will produce a video named output.mp4 that corresponds to the output video.
