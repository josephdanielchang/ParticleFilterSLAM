# USEFUL INFO
# COM z = 0.93m
# Head 0.33m above COM
# Kinect 0.07m above head
# LIDAR 0.015m above head, 1.275m above ground
#
# NOTE
# matrix dimensions in comments refer to dataset 0

from load_data import *
from p2_utils import *
import mapping
import particle_filter
import numpy as np
import matplotlib.pyplot as plt;
from scipy import io
from PIL import Image
import cv2
import os

def test(lidar_data):
    theta = np.arange(-135,135.25,0.25)*np.pi/float(180)
    i=0 #first scan
    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, lidar_data[i])
    ax.set_rmax(10)
    ax.set_rticks([2,4])           # less radial ticks
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    ax.set_title("Lidar scan data", va='bottom')
    plt.show()

# CREATE DIRECTORIES
if not os.path.exists('output_wall'):
    os.makedirs('output_wall')
if not os.path.exists('output_map'):
    os.makedirs('output_map')
if not os.path.exists('output_texture'):
    os.makedirs('output_texture')

# LOAD DATA
dataset = 0
print('get_joint')
joint = get_joint('joint/train_joint%d'%dataset)
print('get_lidar')
lidar = get_lidar('lidar/train_lidar%d'%dataset)
print('getExtrinsics_IR_RGB')
exIR_RGB = getExtrinsics_IR_RGB()
print('getIRCalib')
IRCalib = getIRCalib()
print('getRGBCalib')
RGBCalib = getRGBCalib()
print('get_rgb')
rgb_ts_obj = open('cam/RGB_%d/timestamp.txt'%dataset,'r') 
print('get_depth')
depth_ts_obj = open('cam/DEPTH_%d/timestamp.txt'%dataset,'r')

# Store specific data except rgb_image, depth_depth
# joint_ts, joint_head_angles, lidar_ts, lidar_delta_pose, lidar_scan, rgb_ts, depth_ts, joint_head_angles_matched

joint_ts = joint['ts'].T                      # 38318x1   timestamp
joint_head_angles = joint['head_angles'].T    # 38318x2   neck, head angles

lidar_length = len(lidar)
lidar_ts = np.zeros(lidar_length)
lidar_delta_pose = np.zeros((lidar_length, 3))
lidar_scan = np.zeros((lidar_length, len(lidar[0]['scan'][0])))
for i in range(lidar_length):
    lidar_ts[i] = lidar[i]['t']                      # 12048x1    lidar timestamp              
    lidar_delta_pose[i] = lidar[i]['delta_pose'][0]  # 12048x3    odometry change in world frame
    lidar_scan[i][:] = lidar[i]['scan']              # 12048x1081 lidar scan                   

rgb_length = sum(1 for line in open('cam/RGB_%d/timestamp.txt'%dataset))
rgb_ts = np.zeros(rgb_length)
depth_ts = np.zeros(rgb_length)
counter = 0
for line in rgb_ts_obj:
    rgb_ts[counter] = line.split()[1]   # 228x1 rgb timestamp
    counter += 1
counter = 0 
for line in depth_ts_obj:
    depth_ts[counter] = line.split()[1] # 228x1 depth timestamp
    counter += 1
print('specific data loaded')  

# PREPROCESSING
# remove scan points too close <0.5m, far >20m, hit ground
print('remove_bad_data')
del_count = 0                               # increments as frames deleted
for i in range(lidar_length):               # 12048
    if i == lidar_length-del_count:
        break
    for (k,v) in enumerate(lidar_scan[i]):  # 12048
        if v > 20 or v < 0.5:
            lidar_scan = np.delete(lidar_scan, np.s_[i-del_count], axis=0)  # delete ith row of lidar_scan
            lidar_ts = np.delete(lidar_ts, np.s_[i-del_count], axis=0)      # delete ith row of lidar_scan
            lidar_delta_pose = np.delete(lidar_delta_pose, np.s_[i-del_count], axis=0)  # delete ith row of lidar_scan
            del_count += 1
            break
        
        # find laser hitting points in lidar frame, transform to world frame, removes points z close to 0
        if joint_head_angles[i][1] > 0:
            lidar_scan_w_z = 1.275 - lidar_scan[i][k] * np.sin(joint_head_angles[i][1])
        elif joint_head_angles[i][1] < 0:
            lidar_scan_w_z = 1.275 + lidar_scan[i][k] * np.sin(joint_head_angles[i][1])
        # else:
        #   l0_scan_w_z = 1.275
        if lidar_scan_w_z < 0:
            lidar_scan = np.delete(lidar_scan, np.s_[i-del_count], axis=0)  # delete ith row of lidar_scan
            lidar_ts = np.delete(lidar_ts, np.s_[i-del_count], axis=0)      # delete ith row of lidar_scan
            lidar_delta_pose = np.delete(lidar_delta_pose, np.s_[i-del_count], axis=0)  # delete ith row of lidar_scan
            del_count += 1
        break

lidar_length = len(lidar_scan)  # recompute lidar length after preprocessing

# MATCH JOINT LIDAR TIMESTAMPS
# averages 2 closest joint timestamps and averages for new lidar data
print('match joint lidar')
joint_head_angles_matched = np.zeros((lidar_length, 2))   # 11987x2 neck, head angles
for i in range(lidar_length):
    diff = abs(joint_ts - lidar_ts[i])
    idx1 = np.argmin(diff)              # first closest number
    diff[idx1] = 10000
    idx2 = np.argmin(diff)              # second closest number
    joint_head_angles_matched[i] = (joint_head_angles[idx1][:]+joint_head_angles[idx2])/2

# DISPLAY FIRST SCAN
print('test')
test(lidar_scan)
    
# INITIALIZE MAP
# units in meters, initially set p(occupied)=0.5 with log-odds=0
print('initialize map')
MAP = {}
MAP['res'] = 0.05
MAP['xmin'] = -30
MAP['ymin'] = -30
MAP['xmax'] = 30
MAP['ymax'] = 30
MAP['sizex'] = int(np.ceil((MAP['xmax']-MAP['xmin'])/MAP['res']+1)) 
MAP['sizey'] = int(np.ceil((MAP['ymax']-MAP['ymin'])/MAP['res']+1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float32)     # log-odds map
texture_map = np.zeros((MAP['sizex'],MAP['sizey'],3),dtype=np.uint8)    # texture map
texture = 1                                                             # do texture map

# TRANSFORM: LIDAR TO BODY FRAME
bTl = np.array([[1,0,0,0.29833],[0,1,0,0],[0,0,1,0.51435],[0,0,0,1]])

# TRANSFORM: TEXTURE MAPPING
K = np.array([[585.05108211, 0, 242.94140713],[0, 585.05108211, 315.83800193],[0, 0 ,1]])
invK = np.linalg.inv(K)
roll = 0
pitch = 0.36
yaw = 0.021 
Rx = np.array([[1, 0, 0],[0, np.cos(roll), -np.sin(roll)],[0, np.sin(roll), np.cos(roll)]])
Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw),  np.cos(yaw), 0], [0, 0, 1]])
bTi_p = np.array([0.18, 0.005, 0.36]).reshape(3,1)
bTi_R = np.dot(np.dot(Rz,Ry),Rx)
bTi = np.block([ [bTi_R, bTi_p],[np.zeros((1,3)), 1] ])

# INITIALIZE PARTICLES
# in world frame, initially equal to robot pose
print('initialize particles')
num_p = 40                                           # number of particles
Sw = np.zeros((3,num_p))                             # particle state Sw = [x, y, theta]
weight = np.array([1/num_p]*num_p).reshape(1,num_p)  # intialize weights to 1/num_p

# INITIALIZE TRAJECTORIES
print('initialize trajectories')
trajectory = np.array([[0],[0]])

for i in range(lidar_length):

    # PLOT RESULTS
    if (i%100==0):
        print('plot iteration'+str(i))
        # recover map pmf from log-odds
        output_map = ((1-1/(1+np.exp(MAP['map']))) < 0.1).astype(np.int)
        output_wall = ((1-1/(1+np.exp(MAP['map']))) > 0.9).astype(np.int)

        # convert laser end point to world frame grid units
        xtraj = np.ceil((trajectory[0,:] - MAP['xmin'])/MAP['res']).astype(np.int16) - 1 
        ytraj = np.ceil((trajectory[1,:] - MAP['ymin'])/MAP['res']).astype(np.int16) - 1
        idxGood = np.logical_and(np.logical_and(np.logical_and((xtraj > 1), (ytraj > 1)), (xtraj < MAP['sizex'])), (ytraj < MAP['sizey']))
        output_map[xtraj[idxGood],ytraj[idxGood]] = 2
        output_wall[xtraj[idxGood],ytraj[idxGood]] = 2
        texture_map[xtraj[idxGood],ytraj[idxGood],:] = np.array([255,0,0])
        plt.imsave('output_wall/'+str(i)+'.png', output_wall)
        plt.imsave('output_map/'+str(i)+'.png', output_map)
        plt.imsave('output_texture/'+str(i)+'.png', texture_map)

    # PREDICTION
    Sw = particle_filter.prediction(Sw, lidar_delta_pose[i])
    
    # GET CURRENT VALID SCAN 
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0        # angles in rad
    ranges = lidar_scan[i,:]                                # next scan
    idxValid = np.logical_and((ranges < 20),(ranges > 0.5)) # valid range
    ranges = ranges[idxValid]                               # keep scan indices with valid range
    angles = angles[idxValid]                               # keep angle indices with valid angle

    # UPDATE PARTICLE FILTER WEIGHTS
    Sw, weight = particle_filter.update(MAP, Sw, weight, ranges, angles, bTl)
    
    # UPDATE MAP
    bestidx = np.argmax(weight)                                 # find index of best particle weight
    xt = Sw[:,bestidx]                                          # find pose of best particle
    trajectory = np.hstack((trajectory,xt[0:2].reshape(2,1)))   # update trajectory
    MAP = mapping.update_map(MAP, xt, ranges, angles, bTl)      # update log-odds map

    # TEXTURE MAPPING
    if (texture):
        diff = abs(depth_ts - lidar_ts[i])
        idx1 = np.argmin(diff)
        diff = abs(rgb_ts - lidar_ts[i])
        idx2 = np.argmin(diff)
        
        depth = cv2.imread('cam/DEPTH_%d/%d.png'%(dataset,(idx1+1)),-1)    # load depth
        rgb = cv2.imread('cam/RGB_%d/%d.jpg'%(dataset,(idx2+1)))           # load rgb
        rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
        
        texture_map = mapping.texture(rgb, depth, bTi, xt, texture_map, invK, MAP)

    # RESAMPLE PARTICLES
    Neff = 1/np.dot(weight.reshape(1,num_p), weight.reshape(num_p,1))
    if Neff < 5:
        Sw, weight = particle_filter.stratified_resample(Sw, weight, num_p)


print('last iteration')
output_map = ((1-1/(1+np.exp(MAP['map']))) < 0.1).astype(np.int)
output_wall = ((1-1/(1+np.exp(MAP['map']))) > 0.9).astype(np.int)

# convert laser end point to world frame grid units
xtraj = np.ceil((trajectory[0,:] - MAP['xmin']) / MAP['res'] ).astype(np.int16) - 1 
ytraj = np.ceil((trajectory[1,:] - MAP['ymin']) / MAP['res'] ).astype(np.int16) - 1
idxGood = np.logical_and(np.logical_and(np.logical_and((xtraj > 1), (ytraj > 1)), (xtraj < MAP['sizex'])), (ytraj < MAP['sizey']))
output_map[xtraj[idxGood],ytraj[idxGood]] = 2
output_wall[xtraj[idxGood],ytraj[idxGood]] = 2
plt.imsave('output_wall/'+str(i)+'.png', output_wall)
plt.imsave('output_map/'+str(i)+'.png', output_map)
plt.imsave('output_texture/'+str(i)+'.png', texture_map)














