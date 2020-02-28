# ECE276A Project 2: Particle Filter SLAM

## Files

load_data.py
- load data, display data

main.py
- load data, preprocessing, calcaulate scan, wall, texture maps

make_video.py
- load result images, create video

mapping.py
- update log-odds map, update texture map

p2_utils.py
- map correlation, bresenham ray tracing

particle_filter.py
- predict particle pose, update particle weights, resample

## Folders

output_map
- contains saved output scan maps

output_wall
- contains saved output wall maps

output_texture
- contains saved output texture maps 

cam
- rgb, depth data

joint
- joint data containing head and neck angles

lidar
- laser scan data, robot's delta pose

results_images
- all output images of dataset 0, 3, 4

results_video
- all output video of dataset 0, 3, 4



