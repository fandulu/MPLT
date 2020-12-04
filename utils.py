import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
import cv2
import copy


def get_image(video_path, frame):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC,int(frame)/30*1000)
    success,img = cap.read()
    cap.release()
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #obtain the patch size
    h, w, _ = img.shape
    h = h//2
    w = w//3

    # crop each patch
    img_left = img[:h,:w,:]
    img_front = img[:h,w:w*2,:]    
    img_right = img[:h,w*2:,:]    
    img_back = img[h:,w:w*2,:]
    img_back = np.rot90(img_back,1)

    h = 512
    w = 800

    Images = {}    
    Images['front'] = cv2.resize(img_front, (w,h), interpolation = cv2.INTER_LINEAR)
    Images['left'] = cv2.resize(img_left, (w,h), interpolation = cv2.INTER_LINEAR)
    Images['right'] = cv2.resize(img_right, (w,h), interpolation = cv2.INTER_LINEAR)
    Images['back'] = cv2.resize(img_back[:,::-1,:], (w,h), interpolation = cv2.INTER_LINEAR)
    
    return Images



def plot_pano(pano_locations, img_front, img_left, img_right, img_back):
    
    colors = ['#C0C0C0','#000000','#FF0000','#800000','#FFFF00','#808000','#00FF00','#008000','#00FFFF','#008080',
             '#0000FF','#000080','#FF00FF','#800080','#FF8C00','#C71585','#778899']
    
    fig,ax = plt.subplots(figsize=(7,7))
    plt.axis([-10, 10, -10, 10])
    ax.tick_params(direction='in', colors='r', grid_color='r', pad=-20)
    
    ax.plot([10,-10],[-10,10],'--', c=(0.5, 0.5, 0.5), linewidth=1)
    ax.plot([-10,10],[-10,10],'--', c=(0.5, 0.5, 0.5), linewidth=1)
    
    for i in range(1,6):
        circle = Circle((0, 0), i*2, facecolor='none', edgecolor=(0.5, 0.5, 0.5), linewidth=1, alpha=0.5)
        ax.add_patch(circle)
        
    H, W, _ = img_back.shape
    for i in range(len(pano_locations)):
        ax.scatter(pano_locations[i,1],pano_locations[i,2], c=colors[i%17])
        ax.annotate(i+1, (pano_locations[i,1],pano_locations[i,2]),fontsize=15)
        
        font=cv2.FONT_HERSHEY_SIMPLEX
        if pano_locations[i][0]=='front':
            cv2.putText(img_front, str(i+1), (int(pano_locations[i][4][0]+(pano_locations[i][4][2]-pano_locations[i][4][0])/4), int(pano_locations[i][4][1])), font, 2, (255, 0, 0), 4, cv2.LINE_AA)
        if pano_locations[i][0]=='right':
            cv2.putText(img_right, str(i+1), (int(pano_locations[i][4][0]+(pano_locations[i][4][2]-pano_locations[i][4][0])/4), int(pano_locations[i][4][1])), font, 2, (255, 0, 0), 4, cv2.LINE_AA)
        if pano_locations[i][0]=='back':
            cv2.putText(img_back, str(i+1), (W-int(pano_locations[i][4][2]+(pano_locations[i][4][2]-pano_locations[i][4][0])/4), int(pano_locations[i][4][1])), font, 2, (255, 0, 0), 4, cv2.LINE_AA)
        if pano_locations[i][0]=='left':
            cv2.putText(img_left, str(i+1), (int(pano_locations[i][4][0]+(pano_locations[i][4][2]-pano_locations[i][4][0])/4), int(pano_locations[i][4][1])), font, 2, (255, 0, 0), 4, cv2.LINE_AA)
     
            
    ax.scatter(0, 0, s=400,marker='+', c='r')  

    newax = fig.add_axes([0.305, 0.88, 0.408, 0.4], anchor='NE', zorder=-1)
    newax.imshow(img_front)
    newax.axis('off')

    newax = fig.add_axes([-0.278, 0.3, 0.4, 0.4], anchor='NE', zorder=-1)
    newax.imshow(img_left)
    newax.axis('off')

    newax = fig.add_axes([0.9, 0.3, 0.4, 0.4], anchor='NE', zorder=-1)
    newax.imshow(img_right)
    newax.axis('off')

    newax = fig.add_axes([0.305, -0.27, 0.408, 0.4], anchor='NE', zorder=-1)
    newax.imshow(img_back)
    newax.axis('off')
    
    plt.show()

    
def plot_pano_tracking(frame, pano_locations, pano_bridge, Images):
    
    colors = ['#C0C0C0','#000000','#FF0000','#800000','#FFFF00','#808000','#00FF00','#008000','#00FFFF','#008080',
             '#0000FF','#000080','#FF00FF','#800080','#FF8C00','#C71585','#778899']
    
    fig = plt.figure(figsize=(15,12))
    fig.subplots_adjust(hspace=0,wspace=0)
    ax_center = plt.subplot2grid((16, 20), (4, 6), colspan=8,rowspan=8)

    ax_front = plt.subplot2grid((16, 20), (0, 7), colspan=6,rowspan=4)
    ax_right = plt.subplot2grid((16, 20), (6, 14), colspan=6,rowspan=4)
    ax_back = plt.subplot2grid((16, 20), (12, 7), colspan=6,rowspan=4)
    ax_left = plt.subplot2grid((16, 20), (6, 0), colspan=6,rowspan=4)

    ax_front.axis('off')
    ax_right.axis('off')
    ax_back.axis('off')
    ax_left.axis('off')

    
    ax_center.axis([-10, 10, -10, 10])
    ax_center.tick_params(direction='in', colors='r', grid_color='r', pad=-20)
    
    #draw top-view
    ax_center.plot([10,-10],[-10,10],'--', c=(0.5, 0.5, 0.5), linewidth=1)
    ax_center.plot([-10,10],[-10,10],'--', c=(0.5, 0.5, 0.5), linewidth=1)
    
    for i in range(1,6):
        circle = Circle((0, 0), i*2, facecolor='none', edgecolor=(0.5, 0.5, 0.5), linewidth=1, alpha=0.5)
        ax_center.add_patch(circle)
        
    ax_center.scatter(0, 0, s=400,marker='+', c='r') 

    for i in range(len(pano_locations)):
        ax_center.scatter(pano_locations[i][0],pano_locations[i][1], c=colors[int(pano_locations[i][2]%17)])
        ax_center.annotate(str(int(pano_locations[i][2])), (pano_locations[i][0],pano_locations[i][1]),fontsize=15)

    #draw side-view
    _,W,_ = Images['back'].shape
    img_back_flip = copy.deepcopy(Images['back'][:,::-1,:])
    
    for i in range(len(pano_bridge)):
        font=cv2.FONT_HERSHEY_SIMPLEX
        if pano_bridge[i][0]=='front':
            cv2.putText(Images['front'], str(int(pano_bridge[i][2])), (int(pano_bridge[i][1][0]+(pano_bridge[i][1][2]-pano_bridge[i][1][0])/4), int(pano_bridge[i][1][1])), font, 2, (255, 0, 0), 4, cv2.LINE_AA)
        if pano_bridge[i][0]=='right':
            cv2.putText(Images['right'], str(int(pano_bridge[i][2])), (int(pano_bridge[i][1][0]+(pano_bridge[i][1][2]-pano_bridge[i][1][0])/4), int(pano_bridge[i][1][1])), font, 2, (255, 0, 0), 4, cv2.LINE_AA)
        if pano_bridge[i][0]=='back':
            #cv2.putText(Images['back'], str(int(pano_bridge[i][2])), (int(pano_bridge[i][1][0]+(pano_bridge[i][1][2]-pano_bridge[i][1][0])/4), int(pano_bridge[i][1][1])), font, 2, (255, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img_back_flip, str(int(pano_bridge[i][2])), (W-int(pano_bridge[i][1][2]+(pano_bridge[i][1][2]-pano_bridge[i][1][0])/4), int(pano_bridge[i][1][1])), font, 2, (255, 0, 0), 4, cv2.LINE_AA)
        if pano_bridge[i][0]=='left':
            cv2.putText(Images['left'], str(int(pano_bridge[i][2])), (int(pano_bridge[i][1][0]+(pano_bridge[i][1][2]-pano_bridge[i][1][0])/4), int(pano_bridge[i][1][1])), font, 2, (255, 0, 0), 4, cv2.LINE_AA)

    ax_front.imshow(Images['front'])
    ax_right.imshow(Images['right'])
    ax_back.imshow(img_back_flip)
    ax_left.imshow(Images['left'])

    plt.savefig("outputs/%04d.png" % int(frame))
    plt.show()    

    

def extract_image_patches(image, boxes, corner=False):
    #Input: imgage [h,w,3], boxes [n,5]
    #Output: list n*[patch,box,center_of_image]
    h, w, _ = image.shape
    boxes = np.array(boxes)[:,:4]  
    patches = []
    for box in boxes: 
        x1,y1,x2,y2 = box.astype(int)        
        if x2-x1<30 or y2-y1<30 or x1<5 or x2>w-5:
            continue
        if corner:    
            if not (x1<w/2 and x2>w/2):
                continue     
        patch = image[y1:y2,x1:x2]
        patch_w = x2-x1
        patch_h = y2-y1
        patches.append([patch_h, x1+patch_w/2-w//2, patch_w, box, patch])
    return patches


def get_pano_locations(Locations, d_thred=9):
    pano_locations = []
    for k in list(Locations.keys()):
        for i in range(len(Locations[k])):
            world_depth = Locations[k][i][0]/1000 # from mm to m
            world_center = Locations[k][i][1]/1000 # from mm to m
            world_width = Locations[k][i][2]/1000 # from mm to m
            coor = np.array([world_center,world_depth])

            if k == 'right':
                coor = rotate(coor, -math.radians(90))
            elif k == 'left':
                coor = rotate(coor, math.radians(90))
            elif k == 'back':
                coor = rotate(coor, math.radians(180))
            elif k == '45':
                coor = rotate(coor, -math.radians(45))
            elif k == '135':
                coor = rotate(coor, -math.radians(135))
            elif k == '225':
                coor = rotate(coor, math.radians(135))
            elif k == '315':
                coor = rotate(coor, math.radians(45))   
                
            if abs(coor[0])>d_thred or abs(coor[1])>d_thred:
                pass           
            pano_locations.append([k, coor[0],coor[1], world_width, Locations[k][i][3], Locations[k][i][4]])
            
    pano_locations = np.stack(pano_locations)
    
    return pano_locations


def plot_pano_detection(pano_locations, img_front, img_left, img_right, img_back):
    
    colors = ['#C0C0C0','#000000','#FF0000','#800000','#FFFF00','#808000','#00FF00','#008000','#00FFFF','#008080',
             '#0000FF','#000080','#FF00FF','#800080','#FF8C00','#C71585','#778899']
    
    fig,ax = plt.subplots(figsize=(7,7))
    plt.axis([-10, 10, -10, 10])
    ax.tick_params(direction='in', colors='r', grid_color='r', pad=-20)
    
    ax.plot([10,-10],[-10,10],'--', c=(0.5, 0.5, 0.5), linewidth=1)
    ax.plot([-10,10],[-10,10],'--', c=(0.5, 0.5, 0.5), linewidth=1)
    
    for i in range(1,6):
        circle = Circle((0, 0), i*2, facecolor='none', edgecolor=(0.5, 0.5, 0.5), linewidth=1, alpha=0.5)
        ax.add_patch(circle)
        
    H, W, _ = img_back.shape
    for i in range(len(pano_locations)):
        ax.scatter(pano_locations[i,1],pano_locations[i,2], c=colors[i%17])
        ax.annotate(i+1, (pano_locations[i,1],pano_locations[i,2]),fontsize=15)
        
        font=cv2.FONT_HERSHEY_SIMPLEX
        if pano_locations[i][0]=='front':
            cv2.putText(img_front, str(i+1), (int(pano_locations[i][4][0]+(pano_locations[i][4][2]-pano_locations[i][4][0])/4), int(pano_locations[i][4][1])), font, 2, (255, 0, 0), 4, cv2.LINE_AA)
        if pano_locations[i][0]=='right':
            cv2.putText(img_right, str(i+1), (int(pano_locations[i][4][0]+(pano_locations[i][4][2]-pano_locations[i][4][0])/4), int(pano_locations[i][4][1])), font, 2, (255, 0, 0), 4, cv2.LINE_AA)
        if pano_locations[i][0]=='back':
            cv2.putText(img_back, str(i+1), (W-int(pano_locations[i][4][2]+(pano_locations[i][4][2]-pano_locations[i][4][0])/4), int(pano_locations[i][4][1])), font, 2, (255, 0, 0), 4, cv2.LINE_AA)
        if pano_locations[i][0]=='left':
            cv2.putText(img_left, str(i+1), (int(pano_locations[i][4][0]+(pano_locations[i][4][2]-pano_locations[i][4][0])/4), int(pano_locations[i][4][1])), font, 2, (255, 0, 0), 4, cv2.LINE_AA)
     
            
    ax.scatter(0, 0, s=400,marker='+', c='r')  

    newax = fig.add_axes([0.305, 0.88, 0.408, 0.4], anchor='NE', zorder=-1)
    newax.imshow(img_front)
    newax.axis('off')

    newax = fig.add_axes([-0.278, 0.3, 0.4, 0.4], anchor='NE', zorder=-1)
    newax.imshow(img_left)
    newax.axis('off')

    newax = fig.add_axes([0.9, 0.3, 0.4, 0.4], anchor='NE', zorder=-1)
    newax.imshow(img_right)
    newax.axis('off')

    newax = fig.add_axes([0.305, -0.27, 0.408, 0.4], anchor='NE', zorder=-1)
    newax.imshow(img_back)
    newax.axis('off')

    plt.show()


def rotate(point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = 0, 0
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy



def get_world_locations(patches):
    world_locations = []
    f_l = 500
    for patch in patches:
        ratio = patch[0]/1000    
        ratio *= 1.2
        depth = f_l/ratio
        rescale_body_center = patch[1]/ratio*0.85
        rescale_body_w = patch[2]/ratio
        world_locations.append([depth,rescale_body_center,rescale_body_w, patch[3], patch[4]])
    return world_locations


def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2)) 



def nms(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")



def soft_nms(dets, score_thr=0.1, iou_thr=0.7, method='linear', sigma=0.5 ):
    """Pure python implementation of soft NMS as described in the paper
    `Improving Object Detection With One Line of Code`_.
    Args:
        dets (numpy.array): Detection results with shape `(num, 5)`,
            data in second dimension are [x1, y1, x2, y2, score] respectively.
        method (str): Rescore method. Only can be `linear`, `gaussian`
            or 'greedy'.
        iou_thr (float): IOU threshold. Only work when method is `linear`
            or 'greedy'.
        sigma (float): Gaussian function parameter. Only work when method
            is `gaussian`.
        score_thr (float): Boxes that score less than the.
    Returns:
        numpy.array: Retained boxes.
    .. _`Improving Object Detection With One Line of Code`:
        https://arxiv.org/abs/1704.04503
    """
    if method not in ('linear', 'gaussian', 'greedy'):
        raise ValueError('method must be linear, gaussian or greedy')

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # expand dets with areas, and the second dimension is
    # x1, y1, x2, y2, score, area
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    retained_box = []
    while dets.size > 0:
        max_idx = np.argmax(dets[:, 4], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        retained_box.append(dets[0, :-1])

        xx1 = np.maximum(dets[0, 0], dets[1:, 0])
        yy1 = np.maximum(dets[0, 1], dets[1:, 1])
        xx2 = np.minimum(dets[0, 2], dets[1:, 2])
        yy2 = np.minimum(dets[0, 3], dets[1:, 3])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (dets[0, 5] + dets[1:, 5] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0

        dets[1:, 4] *= weight
        retained_idx = np.where(dets[1:, 4] >= score_thr)[0]
        dets = dets[retained_idx + 1, :]

    return np.vstack(retained_box)


'''
def extract_image_patches(image, boxes):
    #Input: imgage [h,w,3], boxes [n,5]
    #Output: list n*[patch,box,center_of_image]
    h, w, _ = image.shape
    boxes = np.array(boxes)[:,:4]  
    patches = []
    for box in boxes: 
        x1,y1,x2,y2 = box.astype(int)        
        if x2-x1<30 or y2-y1<30 or x1<5 or x2>w-5:
            continue
        patch = image[y1:y2,x1:x2]
        patches.append([patch, box, w//2])
    return patches

def extract_corner_patches(image, boxes):
    #Input: imgage [h,w,3], boxes [n,5]
    #Output: list n*[patch,box,center_of_image]
    h, w, _ = image.shape
    boxes = np.array(boxes)[:,:4]  
    patches = []
    for box in boxes: 
        x1,y1,x2,y2 = box.astype(int)        
        if x2-x1<30 or y2-y1<30:
            continue
        if not (x1<w/2 and x2>w/2):
            continue
        patch = image[y1:y2,x1:x2]
        patches.append([patch, box, w//2])#(x1+x2)/2]) 
    return patches
'''
