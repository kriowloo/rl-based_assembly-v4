ffmpeg -framerate 1 -pattern_type glob -i "new*.png" -c:v libx264 -pix_fmt yuv420p -movflags +faststart output.mp
