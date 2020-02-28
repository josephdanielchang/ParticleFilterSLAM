import numpy as np  
import p2_utils

# compute softmax values for each set of scores in x
def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

# Sw                    particle state in world frame
# lidar_delta_pose      robot movement in world frame
# predicts particle poses and adds gaussian noise
def prediction(Sw, lidar_delta_pose):
	# gaussian noise
	mu = 0              # mean
	N = np.shape(Sw)[1]
	# particle poses
	x_w = Sw[0,:]
	y_w = Sw[1,:]
	theta_w = Sw[2,:]
	# change in robot pose
	delta_x = lidar_delta_pose[0]
	delta_y = lidar_delta_pose[1]
	delta_theta = lidar_delta_pose[2]
	# new particle poses in world frame
	x_w = x_w + delta_x + np.array([np.random.normal(mu, abs(np.max(delta_x))/10, N)])
	y_w = y_w + delta_y + np.array([np.random.normal(mu, abs(np.max(delta_y))/10, N)])
	theta_w = theta_w + delta_theta + np.array([np.random.normal(mu, abs(delta_theta)/10, N)])

	Sw[0,:] = x_w 
	Sw[1,:] = y_w
	Sw[2,:] = theta_w
	return Sw

# MAP                   MAP structure
# Sw                    particle state in world frame
# weight                particle weights
# ranges/angles         laser scan
# bTl                   laser frame to body frame transformation
# update particle weights according to correlation scores
def update(MAP, Sw, weight, ranges, angles, bTl):
	# grid cells representing walls with 1
	map_wall = ((1-1/(1+np.exp(MAP['map']))) > 0.5).astype(np.int)
	x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) # x index of each pixel on log-odds map
	y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) # y index of each pixel on log-odds map
	# 9x9 grid around particle
	x_range = np.arange(-4*MAP['res'],5*MAP['res'],MAP['res']) # x deviation
	y_range = np.arange(-4*MAP['res'],5*MAP['res'],MAP['res']) # y deviation
  	# laser end point in laser frame physical units
	ex = ranges*np.cos(angles)
	ey = ranges*np.sin(angles)
	# convert end point to homogenized coordinates in body frame
	s_h = np.ones((4,np.size(ex)))
	s_h[0,:] = ex
	s_h[1,:] = ey
	s_h = np.dot(bTl,s_h)
	num_particles = np.shape(Sw)[1]
	correlation = np.zeros(num_particles)
	
	for i in range(num_particles):
		xt = Sw[:,i]
		# body to world transform
		x_w = xt[0]
		y_w = xt[1]
		theta_w = xt[2]
		wTb = np.array([[np.cos(theta_w),-np.sin(theta_w),0,x_w],[np.sin(theta_w),np.cos(theta_w),0,y_w],[0,0,1,0],[0,0,0,1]])
		# transform into world frame
		s_w = np.dot(wTb,s_h)
		ex_w = s_w[0,:]
		ey_w = s_w[1,:]
		Y = np.stack((ex_w,ey_w))
		# calculate correlation
		c = p2_utils.mapCorrelation(map_wall, x_im, y_im, Y, x_range, y_range)
		# find best correlation
		correlation[i] = np.max(c)
		
	# update particle weight
	ph = softmax(correlation)
	weight = weight*ph/np.sum(weight*ph)
	return Sw, weight

# Sw            particle state in world frame
# weight        particle weights
# N             number of particles
# resample particles to prevent particle depletion
def stratified_resample(Sw, weight, N):
	Sw_new = np.zeros((3, N))
	weight_new = np.tile(1/N, N).reshape(1, N)
	j = 0
	c = weight[0,0]
	for k in range(N):
		u = np.random.uniform(0, 1/N)   # uniform distribution
		beta = u + k/N                  # scan each part in the circle
		while beta > c :
			j += 1
			c += weight[0,j]        # increase decision section length
		Sw_new[:,k] = Sw[:,j]           # if beta is smaller than many times, put this repeated particles j in new set
		# k=1, k=2, k=3, may all have same particles X[1]
	return Sw_new, weight_new




