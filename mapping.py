from p2_utils import bresenham2D
import numpy as np
import numpy.matlib

# MAP                   MAP structure
# xt                    pose of best particle [x, y, theta]
# ranges/angles         lidar scan
# bTl 		        lidar frame to body frame transform
# update log-odds map using lidar scan
def update_map(MAP, xt, ranges, angles, bTl):
        # calculate body to world transform wTb
        x_w = xt[0]
        y_w = xt[1]
        theta_w = xt[2]
        wTb = np.array([[np.cos(theta_w),-np.sin(theta_w),0,x_w],[np.sin(theta_w), np.cos(theta_w), 0, y_w],[0,0,1,0],[0,0,0,1]])
        # laser start point in world frame grid units
        sx = np.ceil((x_w - MAP['xmin']) / MAP['res'] ).astype(np.int16) - 1 
        sy = np.ceil((y_w - MAP['ymin']) / MAP['res'] ).astype(np.int16) - 1
        # laser end point in laser frame physical units
        ex = ranges*np.cos(angles)
        ey = ranges*np.sin(angles)
        # convert end point to homogenized coordinates in body frame
        s_h = np.ones((4,np.size(ex)))
        s_h[0,:] = ex
        s_h[1,:] = ey
        s_h[2,:] = 0.51435
        # transform scan to world frame using pose
        s_h = np.dot(wTb,np.dot(bTl,s_h))
        # laser end point in world frame grid units
        ex = s_h[0,:]
        ey = s_h[1,:]
        ex = np.ceil((ex - MAP['xmin']) / MAP['res'] ).astype(np.int16) - 1 
        ey = np.ceil((ey - MAP['ymin']) / MAP['res'] ).astype(np.int16) - 1
       
        for i in range(np.size(ranges)):                                # loop through laser scans
                passed_points = bresenham2D(sx, sy, ex[i], ey[i])       # cells between two points
                trajx = passed_points[0,:].astype(np.int16)
                trajy = passed_points[1,:].astype(np.int16)
                idxGood = np.logical_and(np.logical_and(np.logical_and((trajx > 1), (trajy > 1)), (trajx < MAP['sizex'])), (trajy < MAP['sizey']))
                
                # update log-odds map, increase occupied cell, decrease free cell 
                MAP['map'][trajx[idxGood],trajy[idxGood]] += np.log(1/4)
                if ((ex[i]>1) and (ex[i]<MAP['sizex']) and (ey[i]>1) and (ey[i]<MAP['sizey'])):
                        MAP['map'][ex[i],ey[i]] += 2*np.log(4)

        # clip range to prevent over-confidence
        MAP['map'] = np.clip(MAP['map'],10*np.log(1/4),10*np.log(4))
        return MAP
    
# rgb/depth     rgb/depth image
# bTi           lidar to body transform
# xt            best robot pose to determine wTb
# texture_map   current texture map
# update texture map
def texture(rgb, depth, bTi, xt, texture_map, invK, MAP):
        # calculate body to world transform wTb
	x_w = xt[0]
	y_w = xt[1]
	theta_w = xt[2]
	wTb = np.array([[np.cos(theta_w),-np.sin(theta_w),0,x_w],[np.sin(theta_w),np.cos(theta_w),0,y_w],[0,0,1,0],[0,0,0,1]])

	v = np.matlib.repmat(np.arange(0, depth.shape[0], 1), depth.shape[1], 1).T 
	h = np.matlib.repmat(np.arange(0, depth.shape[1], 1), depth.shape[0], 1) 
	v = v.reshape(1,-1)
	h = h.reshape(1,-1) 
	dd = -0.00304*depth.reshape(1,-1) + 3.31 
	Zo = 1.03/dd
	pixels = np.vstack((h, v, Zo))
	Kpixels = np.dot(invK, pixels) 
	pixel_o = np.vstack(( Kpixels, np.ones((1,Kpixels.shape[1])) ))
	pixel_w = np.dot(np.dot(wTb,bTi), pixel_o)
	pixel_w /= pixel_w[3]
	ground_index = pixel_w[2,:] < 0.2
	
	i, j = pixels[0,ground_index], pixels[1,ground_index] 
	rgbi = (i*526.37 + dd[0,ground_index]*(-4.5*1750.46) + 19276.0)/585.051
	rgbj = (j*526.37 + 16662.0)/585.051
	rgbi, rgbj = np.ceil(rgbi).astype(np.uint16)-1, np.ceil(rgbj).astype(np.uint16)-1
	mx = np.ceil((pixel_w[0,ground_index] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
	my = np.ceil((pixel_w[1,ground_index] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1
	indGood1 = np.logical_and(np.logical_and(np.logical_and((mx > 1), (my > 1)), (mx < MAP['sizex'])),(my < MAP['sizey']))
	indGood2 = np.logical_and(np.logical_and(np.logical_and((rgbi > 0), (rgbj > 0)), (rgbi < depth.shape[1])),(rgbj < depth.shape[0]))
	indGood = np.logical_and(indGood1, indGood2)
	texture_map[mx[indGood],my[indGood],:] = rgb[rgbj[indGood],rgbi[indGood],:]
	return texture_map














    
