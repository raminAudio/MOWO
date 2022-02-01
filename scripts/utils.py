# %% [code] {"jupyter":{"outputs_hidden":false}}
"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2
from matplotlib import cm
import matplotlib.pylab as plt


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image_modified(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    w, h = size
    image = image.resize((h,w))

    return image

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data

def get_random_data_modified(annotation_line, input_shape, random=False, max_boxes=8):

    '''image and bounding box rotation'''
    line = annotation_line.split()
    image = Image.open(line[0])
   
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    
    # resize image and bounding boxes
            
    new_image = letterbox_image(image,input_shape)
    image_data = np.array(new_image)/255.
    
    
    iw ,ih = image.size
    h,w = input_shape

    scaleX = w/iw
    scaleY = h/ih
    
    nw = int(iw*scaleX)
    nh = int(ih*scaleY)
    
    dx = (w-nw)//2
    dy = (h-nh)//2
    
    box[:, [0,2]] = box[:, [0,2]]*scaleX + dx
    box[:, [1,3]] = box[:, [1,3]]*scaleY + dy
    
    box_data = box
#     print("*")
#     plt.imshow(image_data)
    
    total_boxes = np.zeros((max_boxes,5)) # maximum boxes
    np.random.shuffle(box_data)
    j= -1
    for i,box_ in enumerate(box_data): 
        if box_[-1] != 4 : 
            j += 1
            total_boxes[j,:] = box_
        plt.plot((box_[0]+box_[2])/2,(box_[1]+box_[3])/2,'x')
    plt.show()

    print("*")
    augment 
    if random:
        angles = [0,0,0,0,1,2,3,4,5]
        angle  = np.random.choice(angles)
        rotated_image_data,rot_matrix = rotate_image(image_data, angle)

        if len(box_data)>0:
            
            box_list = rotateYolobbox(image_data,rotated_image_data,rot_matrix,box_data)
            bbox_data = np.array(box_list)
            image_data = rotated_image_data
            plt.imshow(image_data)
            
            for box_ in bbox_data: 
                plt.plot((box_[0]+box_[2])/2,(box_[1]+box_[3])/2,'x')
            plt.show()
            
            total_boxes = np.zeros((max_boxes,5))
            

            image  = Image.fromarray(np.uint8(image_data*255)).convert('RGB')

            new_image = letterbox_image(image,input_shape)
            
            image_data = np.array(new_image)/255.
            org_x = np.shape(image_data)[0]
            org_y = np.shape(image_data)[1]
        
            r_x = np.shape(rotated_image_data)[0]
            r_y = np.shape(rotated_image_data)[1]
            
            # cropping the image, taking the diff
            r_x_d = r_x - org_x
            r_y_d = r_y - org_y
            print(r_x_d)
            print(r_y_d)
            image_data = rotated_image_data#[0:org_x,0:org_y,:]
            plt.imshow(image_data)
            

            iw ,ih = image.size
            h,w = input_shape

            scaleX = w/iw
            scaleY = h/ih

            nw = int(iw*scaleX)
            nh = int(ih*scaleY)

            dx = (w-nw)//2
            dy = (h-nh)//2

            box_data[:, [0,2]] = box_data[:, [0,2]]*scaleX + dx
            box_data[:, [1,3]] = box_data[:, [1,3]]*scaleY + dy

            box_data = np.int32(box_data)
            
            for i,box_ in enumerate(box_data): 
#                 box_ = np.array([box_[0] , box_[1] , box_[2] , box_[3], box_[4] ])
                total_boxes[i,:] = box_
                plt.plot((box_[0]+box_[2])/2,(box_[1]+box_[3])/2,'x')
                
            plt.show()
            print("-----")
    return image_data, total_boxes



def rotate_image(image, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    height, width  = image.shape[:2]  # image shape has 3 dimensions
    image_center = (width / 2 , height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    rotation_angle = angle * np.pi / 180
    rot_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origin) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_image = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))

    return rotated_image , rotation_mat#[:,:2]


def rotateYolobbox(image,rotated_image,rot_matrix,box_data):
        
        rshape  = np.shape(rotated_image) # self.rotate_image().shape[:2]
        new_height , new_width   = (rshape[0],rshape[1])
        
        ishape = np.shape(image)
        H , W   = ishape[0],ishape[1]
        
    
        diffH = new_height - H 
        diffW = new_width  - W 
        
        
        new_bbox = []
        for bbox in box_data:
            if len(bbox):
                
                xmin = bbox[0] 
                ymin = bbox[1] 
                xmax = bbox[2] 
                ymax = bbox[3] 
                bbox_= np.array([[xmin,ymin,1],[xmax,ymax,1]]).T

                bb_rotated = np.dot(rot_matrix,bbox_).T
                new_bbox.append([bb_rotated[0][0],bb_rotated[0][1],bb_rotated[1][0],bb_rotated[1][1],bbox[-1]])
                

        return new_bbox

