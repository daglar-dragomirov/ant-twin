# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:25:26 2023

@author: Arthur Matte
@contact auhor: matte.arthur01[at]gmail.com
"""
#### Import library
import cv2
import numpy as np
import pathlib
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import openpyxl
from openpyxl import Workbook
from scipy.spatial import KDTree
import tifffile
import os


#### 
def retrieve_biggest_tif_file_in_directory(directory):
    # Initialize variables to keep track of the biggest file and its size
    biggest_file = None
    biggest_size = 0
    
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a .tiff file
        if filename.endswith('.tif'):
            # Get the full path of the file
            filepath = os.path.join(directory, filename)
            
            # Get the size of the file
            size = os.path.getsize(filepath)
            
            # Check if the current file is bigger than the biggest file found so far
            if size > biggest_size:
                biggest_file = filepath
                biggest_size = size
    
    # Return the path of the biggest .tiff file
    return biggest_file

def build_dictionary(input_folder):
    """
    Build a dictionary from a directory with vial labels as keys and lists of specimen folders as values.

    Parameters:
    - input_folder (str): The path to the input folder.

    Returns:
    - dict: A dictionary with vial labels as keys and lists of specimen folders as values.
    """
    dictionary = {}

    for file in pathlib.Path(input_folder).iterdir():
        # Iterate through all filenames in the folder
        obj = str(file)
        indiv_folder = obj[obj.rfind("\\") + 1:]  # Find the last "\\" in the string

        indiv = indiv_folder[:indiv_folder.find("_z")]

        if indiv not in dictionary:
            dictionary[indiv] = [indiv_folder]
        else:
            dictionary[indiv] = dictionary[indiv] + [indiv_folder]

    for i in dictionary.keys():
        dictionary[i] = sorted(dictionary[i])  # Sort folders for consistency

    # Modify the dictionary to keep only the repeat scans when they exist
    for indiv in dictionary.keys():
        for i in dictionary[indiv]:
            if 'repeat' in i:
                rep = str(i)
                rep = rep[:rep.find('_repeat')]

                if rep in dictionary[indiv]:
                    index = dictionary[indiv].index(rep)
                    del dictionary[indiv][index]

    return dictionary

def open_single_img_tif(file_path, slice_index):
    '''
    Parameters
    ----------
    file_path : path to the 3D tiff image
    slice_index : index of the slice to load

    Returns
    -------
    image : 2D image

    '''
    
    # Read the specific slice using tifffile
    with tifffile.TiffFile(file_path) as tif:
        image = tif.asarray(key=slice_index) # Access the desired slice
    return image


def count_2d_images(tiff_file_path):
    ''' Extract the number of 2D image in a 3D tiff file '''
    try:
        # Read the TIFF file
        with tifffile.TiffFile(tiff_file_path) as tiff:
            # Get the number of pages in the TIFF file
            num_pages = len(tiff.pages)

            # Assuming all pages have the same shape, get the shape of the first page
            if num_pages > 0:
                first_page_shape = tiff.pages[0].shape
                num_2d_images = num_pages if len(first_page_shape) == 2 else 0
            else:
                num_2d_images = 0

            return num_2d_images

    except Exception as e:
        raise ValueError(e)



#### image visualisation

def display(img, time=0, scale_percent=40, title = 'Image' ):
    
    ''' Display the input image with option of rezising and display time'''
    
    if len(img) < 20: img = np.concatenate(img, axis=1)
        
    if not scale_percent == 100 :
        # This function resize the image according a given percentage of its original size 
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # resize image
            
    cv2.imshow(title, img)
    cv2.waitKey(time)
    cv2.destroyAllWindows()

def resize_img(img, scale_percent):
    ''' Resize the input image according a given percentage of its original size '''

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # resize image
        
    return resized


def draw_cnts(image, contours):
    color = (255,255,255)
    stencil = np.zeros(image.shape).astype(image.dtype) #build a black backgroud at the image dimensions
    cv2.fillPoly(stencil, contours, color) #draw a polygone on the image
    return stencil

#### basic operations on contours
def centroid_calculator (contour):
    '''Return the centroid of a given contour'''
    
    M = cv2.moments(contour)
    if not M['m00'] == 0 :
        # Calcule centroid of the contour
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        centro = [cx , cy]
    else: centro = [0,0]
    return centro

def distanceCalculate(p1, p2):
    ''' calculate the distance between two points
    p1 and p2 in format (x1,y1) and (x2,y2) tuples'''
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

def cnt1_inside_cnt2(cnt1, cnt2, min_dist):
    ''' 
    Test if a contour is inside another
    return true if all the points of the cnt1 are inside the cnt2 and distant of an inmput value 
    corresponding to the nb of pixels 
    '''
    
    if len (cnt1) > 15 :
        sampling_size = len(cnt1)/10
        gap = round(len(cnt1)/sampling_size)
    else: gap = 1
    
    for i, point in enumerate(cnt1[::gap]):
        point_x = int(point[0][0])
        point_y = int(point[0][1])        
        dist = cv2.pointPolygonTest(cnt2, (point_x, point_y), True)
        
        if dist < min_dist : break # <10 because sometime external cnt are a bit include in another and should not be removed
    
    if dist < min_dist : return False
    else : return True



def cnt1_partly_in_cnt2(cnt1, cnt2, min_dist_outise):
    
     # This function test if a contour is fully inside another
     # http://www.devdoc.net/linux/OpenCV-3.2.0/d3/dc0/group__imgproc__shape.html#ga1a539e8db2135af2566103705d7a5722
     # return true even if the points of the cnt1 are distant of maximum 10 pixels outside
     
     """ pointPolygonTest outputs:
                 > 1 : inside
                 0 : on a edge
                 < 1 : outside
     """
     
     # fully inside
         # no points are outside
         # if dist < thresh_dist : break ; return False
     # fully outside
         # no points are inside
         # if dist > thresh_dist : break ; return False
     # partially inside
        # at least one point is inside
        # if dist > thresh_dist : break ; return True
     # partially outside 
         # at least one point is outside
         # if dist < thresh_dist : break ; return False
     
     if len (cnt1) > 15 :
         sampling_size = len(cnt1)/10
         gap = round(len(cnt1)/sampling_size)
     else : gap = 1
 
     for i, point in enumerate(cnt1[::gap]):
        point_x = int(point[0][0])
        point_y = int(point[0][1])
        dist = cv2.pointPolygonTest(cnt2, (point_x, point_y), True)
        if dist > min_dist_outise : break # <10 because sometime external cnt are a bit include in another and should not be removed

     if dist > min_dist_outise: return True
     else : return False





def black_insid_hull(image, contour):  
    
    '''This function test if a contour contain black pixel in its internal area'''
    
    # usefull link: https://realpython.com/python-bitwise-operators/
    
    hull = cv2.convexHull(contour)
        
    stencil = np.zeros(image.shape).astype(image.dtype) #build a black backgroud at the image dimensions
    stencil [ : , :] = 255 # fill the stencil as withe
    
    color = 0

    cv2.fillPoly(stencil, [hull], color) #draw a filled polygone on the image
    result = cv2.bitwise_or(image, stencil)
    
    if np.sum(result)/ 255 < image.shape[0] * image.shape[1] - 50 : return True
    else : return False


#### image proceesing

def segmente_image(image, thresholds):  
    
    # This function apply a mask on the image to keep only the pixel within a given range of color values
    
    lower_bound = int(thresholds[0])
    upper_bound = int(thresholds[1])
    
    mask = cv2.inRange(image, lower_bound, upper_bound)
    
    # Sharpening the image edges
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    image_sharp = cv2.filter2D(src=mask, ddepth=-1, kernel=kernel)
    
    # Reduce the noise in the mask
    mask_adj = cv2.bilateralFilter(image_sharp, 
                                    d = 5, 
                                    sigmaColor = 75, 
                                    sigmaSpace = 75,
                                    borderType = cv2.BORDER_ISOLATED)
    
    return mask_adj


def color_image_processing(image):   
    
    ''' This function improve some parameters of the image to facilitat the furter segmentation

    Ussefull links:
        Reduce blur by keeping edges : https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
        About contrast and brightness : https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
    '''
    
    # Sharpening the image edges
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    

    # Reduce blur by keeping edge 
    adjusted_blur = cv2.bilateralFilter(image_sharp, 
                                        d = 6,           # Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace. 
                                        sigmaColor = 75, # A larger value of the parameter means that farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color. 
                                        sigmaSpace = 75, # A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor )
                                        borderType = cv2.BORDER_ISOLATED)
    
    # Adjust contraste and brightness
    # alpha = 1.3 # Contrast control (1.0-3.0)
    # beta = 0   # Brightness control (0-100)
    # adjusted = cv2.convertScaleAbs(adjusted_blur, alpha=alpha, beta=beta)
        
    return adjusted_blur

def color_image_processing_2(image):   
    
    # This function improve some parameters of the image to facilitat the furter segmentation
    
    # ussefull links:
        # Reduce blur by keeping edges : https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
        # About contrast and brightness : https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape

    # Reduce blur by keeping edge 
    adjusted_blur = cv2.bilateralFilter(image, 
                                        d = 6, 
                                        sigmaColor = 75, 
                                        sigmaSpace = 75,
                                        borderType = cv2.BORDER_ISOLATED)
    
    # Sharpening the image edges
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    image_sharp = cv2.filter2D(src=adjusted_blur, ddepth=-1, kernel=kernel)
    

    # Reduce blur by keeping edge 
    adjusted_blur = cv2.bilateralFilter(image_sharp, 
                                        d = 6, 
                                        sigmaColor = 75, 
                                        sigmaSpace = 75,
                                        borderType = cv2.BORDER_ISOLATED)
    
    # Adjust contraste and brightness
    # alpha = 1.3 # Contrast control (1.0-3.0)
    # beta = 0   # Brightness control (0-100)
    # adjusted = cv2.convertScaleAbs(adjusted_blur, alpha=alpha, beta=beta)
        
    return adjusted_blur

def get_metrics(binary_image):

    ''' This fucntion return the number of white pixels in image of grey scale (i.e. white = 255) '''
    
    return np.sum(binary_image)/255

#### remove vial


def remove_vial(image, croping_strength = None):
    
    '''  This funciton identify the vial on the image and remove it 
    
    The input is a single chanel (grey scale) image and value beween 0 and 1 indication the croping strength 
    
    This function work on the idea that there is a strong contrast between the black backround and the vial & its content.
    By applying a very weak threshold, we segregate the black background and the vial & its content in a binarized image.
    
    Then we identify where is the border and, assuming the circularity of the vial, we crop the image with a circle shape.
        The croping_strength refer to how much the circle radius will be decreased from the original border finded. 
        A value of 1 will corespond to the extrem border, while a decreasing value will reduce the radius of the final circle.
    
    
    
    This function is not always 100% effective althougt work well in most of case. 
        Sometime, part of the vial remains
        When the specimen is very close to the vial, part of it can be removed
        This function doesn't work proprely for specimens scanned with a magnification of 2. '''
    

    if croping_strength == None: croping_strength = 0.96

    # Apply a strong threshold to binarize the image
    ret, thresh = cv2.threshold(image, 40,255,cv2.THRESH_BINARY) # Apply a strong threshold to binarize the image
    
    # Detect contours on the binarized image
    ext_contours, hierarchy_ext = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contour in it
    ext_contours = list(ext_contours)
       
    # Get image parameters for further comparison
    height_img, width_img = image.shape
    area_img = height_img * width_img 
    
    # Filter the detected contours to keep only contours enought bigger to be the vial
    relevant_cnts = []
    for cnt in ext_contours:
        area = cv2.contourArea(cnt)
        if area/area_img > 0.1 : relevant_cnts.append(cnt) # Assume that the vial is not smaler than 10 times the size of the image
    
    
    # Keep only the biggest contour finded among the potential vial's contours
    if len (relevant_cnts) > 0 : 
        
        # Build the list of the contour area
        area_list = []
        for cnt in relevant_cnts: area_list.append(cv2.contourArea(cnt))
        
        # Identify which contour is the biggest   
        circle = relevant_cnts[area_list.index(max(area_list))]
        
        # Get parameter of the biggest contours, considering it as a circle
        a, b = centroid_calculator(circle) # calculate center
        r = (max(area_list) / math.pi )**0.5 # calculate radius
        
        # Build the final image
        stencil = np.zeros(image.shape).astype(image.dtype) # Build a black backgroud at the image dimensions
        cv2.circle(stencil, (a, b), round(r*croping_strength), (255, 255, 255), -1) # Draw the circle filled in white. Note that the applied radius is a bit lower that the detected one, to actually crop the image.
        result = cv2.bitwise_and(image, stencil) # Remplace the white part by the original image content
         
        return result
    
    else: return image # If no potential vial are detected, the original image is returned



def remove_vial_2(image, ratio_vial_image):
       
    # The input should be a binarized image
   
    # Given that the function "remove_vial()" doesn't work properly for specimens scanned with a magnification of 2,
    # I built another way to remove the vial. I must admit that this one is very promising although might require some additional fine-tunes
       
    # This function rely on three characteristics of the vial to detect it and remove it.
       	# The vial is circular. As such, we can expect that the ellipse fitted on its belonging contours will be circular as well
       	# The center of the vial is close to the center of the image
       	# The internal area of the vial represents a substantial amount in the image (not always true, this is why you can fine-tune it by giving it in input as ratio_vial_image. A common value is 0.25)
       
    # Considering that the vial could appear as partial, this function relies on the fitting of ellipses on contours to estimate if they are likely to belong to the vial or not.
    # The partial contours leading to an ellipse bigger than the image are painted in black
    # The smallest ellipse identified as belonging to the vial is used to crop the image on a circular shape as did the function "remove_vial()"
    # The idea is therefore to identify a contour which is the vial or part of the vial, and crop arround
       
    # This function is not always 100% effective although work well in most case.
       	# The most common error is to don't identifying the contours belonging to the vial
       	# Rarely, very big contours having a circular shape can be interpreted as the vial, leading to crop wrongly the image.
    
    # I personally used this function in addition to the function "remove_vial", after the step of the threshold segmentation.
    # You can easily use it by applying, first of all, a mask on the image using the following line:    
       	# mask = cv2.inRange(image, lower_bound, upper_bound)
       	# I would suggest lower_bound = 128 and upper_bound = 255.

    
    min_vial = list()
    min_vial_area = image.shape[0]*image.shape[1] # Estimate parameter of the image for further comparison
     
    # remove the blur, see the function color_image_processing() for details
    adjusted_blur = cv2.bilateralFilter(image, 
                                d = 6, 
                                sigmaColor = 75, 
                                sigmaSpace = 75,
                                borderType = cv2.BORDER_ISOLATED)
    
    # Detect contours on the binarized image
    contours, hierarchy = cv2.findContours(adjusted_blur.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) 
    contours = list(contours)
    
    
    if len(contours) > 0 :
        
        # Test if any detected contours fit the assumptions to be reasonably considered part of the vial. 
        for i in range(len(contours)):
            
            current_contour = contours[i]
            
            if not cv2.arcLength(current_contour,True) < 100  :  # Ignore too litle contours
                
                # Fit an elipse arround the contour
                (x, y), (MA, ma), angle = cv2.fitEllipse(current_contour) 
                
                # Get parameters of the elipse
                dist = distanceCalculate((x, y), (image.shape[0]/2, image.shape[1]/2)) # Calculate the distance between the center of the elipse and the center of the image
                area_elipse = MA/2*ma/2*math.pi # Calculate area of the ellisp
                # Test if the fitted elipse is actually close to be a circle
                # Test if the center of the fitted elipse is close to the center of the image 
                # Test if the area of the fitted elipse actually represent a substantial* amount in the image (*given in input)
                if min(MA, ma) > 0 and (max(MA, ma)/min(MA, ma)) < 1.09  and dist < image.shape[0]/8 and area_elipse > (image.shape[0]*image.shape[1])* ratio_vial_image: 
  
                    # To keep the smallest elipse fiting these conditions
                    if area_elipse < min_vial_area : 
                        
                        min_vial_area = area_elipse
                        min_vial = current_contour
            
                # If the elipse is bigger than the image, so the contour associated to this elipse is paint in black to remove it from the image.
                elif area_elipse > (image.shape[0]*image.shape[1]) :  cv2.drawContours(image, current_contour, -1, (0, 0, 0), 10)  # Paint the ctn in black
                
    # Crop the image
    if len(min_vial) > 1 : # If a ellips fiting the potential vial is detected
    
        # get the ellips of the min circle ID
        (x, y), (MA, ma), angle = cv2.fitEllipse(min_vial)
        
        # Build the final image
        stencil = np.zeros(image.shape).astype(image.dtype) #build a black backgroud at the image dimensions
        cv2.ellipse(stencil, (int(x), int(y)), (int(MA/2*0.9), int(ma/2*0.9)), angle, 0, 360,  (255, 255, 255), -1) # Draw a circle filled in white associated with the ellips parameters. Note that the applied radius is a bit lower that the detected one, to actually crop the image.
        result = cv2.bitwise_and(image, stencil) # Remplace the white part by the original image content
        
        return result
    else :return image # If no such contours are considered relevant, the original image is returned



def remove_wings(segmented_filtered, thresh_len = 80, thresh_defects = 20, dilate_size = 9) :
    
    dilated = dilate(segmented_filtered, dilate_size)

    contours, hierarchy = retrieve_contours(dilated, 'all')
    parent_cnts, child_cnts, orphan_cnts, ambiguous_parent_cnts = classify_cnts(contours, hierarchy)

    error_cnt = []
    if orphan_cnts is not None and len(orphan_cnts) > 0:
            
        for contour in orphan_cnts : 
            
            if len(contour) < thresh_len: 
                error_cnt.append(contour)
                continue
            
            hull = cv2.convexHull(contour, returnPoints=False)
            # Calculate convexity defects
            try:
                defects = cv2.convexityDefects(contour, hull)
            except cv2.error as e:
                # error_cnt.append(contour)
                # black_image = np.zeros(segmented_filtered.shape, dtype=np.uint8)
                # cv2.drawContours(black_image, [contour], -1, 255, -1) 
                # display(black_image, 0, 30)
                continue  
                   
            
            # Extract values from column 4
            column_4_values = defects[:, 0, 3]
            max_column_4 = np.max(column_4_values)
            pruned_column_4 = column_4_values[column_4_values < max_column_4]
            average_pruned_column_4 = np.mean(pruned_column_4)
            
            ratio = max_column_4 / average_pruned_column_4
            
            if ratio < thresh_defects : error_cnt.append(contour)

    pruned_img = segmented_filtered.copy()
    if len(error_cnt) > 0 : cv2.drawContours(pruned_img, error_cnt, -1, 0, -1) 
    
    return pruned_img



#### begining and end 

def contour_in_the_image(image, thresholds):
    
    '''This function test whether an image contains contours belonging to the specimen or not.
    
    The input is a single-channel (grayscale) image and return True or false according the result. 
    
    This function segments the image with common threshold values and tests whether the image contains at least one contour with a perimeter larger than 50 pixels.
    
    '''
    
    # Fisrtly, reduce the blur
    adjusted_blur = cv2.bilateralFilter(image, 
                                        d = 15, 
                                        sigmaColor = 75, 
                                        sigmaSpace = 75,
                                        borderType = cv2.BORDER_ISOLATED)
    
    # Apply a threshold with common values for fast segmentation
    lower_bound = int(thresholds[0])
    upper_bound = int(thresholds[1])
    
    mask = cv2.inRange(adjusted_blur, lower_bound, upper_bound)
    
    # Retrieve the contours on the binarized image
    all_contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    CNT = False # By default, there is no contour
    for cnt in all_contours:
        if cv2.arcLength(cnt,True) > 50 : CNT = True ; break # If at least one contour have a perimeter of more than 50 pixels, we considert that the image contain a contour
    
    return CNT



def beginning_and_end(file_path, segmentation_parameters, accuracy = 30):
    
    '''This function detects the beginning and end of the individual within the image stack.
    
    The input is a stack of 2D single-channel (grayscale) images.
    The output is the index of the beginning and the end of the individual in the stack (± accuracy required images, always including the individual).
    The accuracy define the number of image between the analysed image. Higher value will make the function faster but less accurate.
    If na value is specified, the value will be 30
    
    This function uses a fast segmentation function to evaluate if a picture contains the cuticle contour or not.
    It then compares the overall occurrence of contours across the stack to figure out where the individual is.
    '''
    
    threshold, _, vial_effort, _ = segmentation_parameters

    # Prepare the environment    
    beginning = None
    end = None
    
    presence_of_cnt = []
    index = []
    
    len_images = count_2d_images(file_path)
    
    # Retrieve the presence of contours accross the stack
    for i in tqdm (range(len_images), desc="Find extremities…", ascii=False, ncols=75): 
        
        if i % accuracy == 0 : # Analyse one image per slice of 30
        
            image = open_single_img_tif(file_path, i)

            adjusted_image = color_image_processing(image) # Process image colors
            
            without_vial = remove_vial(adjusted_image) # Remove the vial contours if finded
            
            if vial_effort :
                segmented = segmente_image(without_vial, threshold) # Segmmente image according the threshold
                without_vial = remove_vial_2(segmented, vial_effort) # Remove the vial contours if finded
                
            presence_of_cnt.append(contour_in_the_image (without_vial, threshold)) # Retrieve the presenc eof contour in it or not. 
            
            index.append(i) # Usefull in the futur 
            
    
    # Visualize the distribution
    # plt.plot(index, presence_of_cnt)
    
    # Identify the begening and the end of the specimens in the slices
    if not False in presence_of_cnt : beginning = 0 ; end = len_images # For the case where their is contour everywhere
    else :    
        
         # To find the beginning the program will start from one extermity toward the other and stop when it find occurence of contours 5 times succesively.
         # To find the end the program will start from the other extermity toward the other and stop when it find occurence of contours 5 times succesively.
         # Use the recurent presence of contours 5 time succesively allow to accont for potential false positive in contour detection
         
        
        # Find the beginning
        for i in range(len(presence_of_cnt)-5):
            
            if presence_of_cnt[i:i+5] == [True, True, True, True, True] : 
                
                if i == 0: beginning = index[0]; break
                else : beginning = index[i-1]; break
        
        # Find the end
        for i in range(len(presence_of_cnt), 6, -1):   
            if presence_of_cnt[i-5:i] == [True, True, True, True, True] : 
                if i == len(index): end = index[-1]; break
                else : end = index[i]; break
        
        
        # Analyse results
        if beginning == None and end == None : # when nothing is find
            beginning = len_images - 1
            end = len_images
        
        # It append sometime, that a noise lead to identify contours corresponding not to the individual. 
        # This would lead to identify a wrong beginning and end. But these cases can be identified as they lead to discontinuity in the occurence of contours. (Visualize the distribution help to understand that point)
        # As such, the program will verify that the beginning and end values make sense regarding the stack and the occurence of contours in it.
        
        else :
            
            # Remove parts indentified as useless 
            presence_of_cnt = presence_of_cnt [ index.index(beginning) : index.index(end) ]
            index = index [ index.index(beginning) : index.index(end) ]
                        
    
            if False in presence_of_cnt [1 : len(presence_of_cnt)-1] : 
                
                # The pruned stack should not contain picture without contour but apparently have.
                # So the lines below find where are the false and keep the longest part without false
                
                potential_beginning = list(reversed(index))[list(reversed(presence_of_cnt[:len(presence_of_cnt)-1])).index(False)+1]
                potential_end = index[presence_of_cnt[1:].index(False)+1] 
            
                # Keep the longest succession of contour occurence
                if potential_end - beginning > end - potential_beginning : end = potential_end
                else : beginning = potential_beginning
        
    return beginning, end


#### Part II
#### estimate body volume

#### basic functions
   
def retrieve_contours(image, method):  
    
    if method == 'all' : contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # retrieve all contours
    elif method == 'external' : contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # retrieve all contours
        
    if len(contours) > 0 : # verify that contours were detected
        contours = list(contours) # Necessary step for the proper fonctionning
        hierarchy = hierarchy[0] # Necessary step for the proper fonctionning
            
    return contours, hierarchy

def classify_cnts(contours, hierarchy):  

    to_del = []
    parent_cnts = []
    ambiguous_parent_cnts = []
    child_cnts = []
    
    if len(contours) < 1:
        return None, None, None, None
    
    for i in range(len(contours)):
        # Vérifier si le contour a un parent
        if hierarchy[i][2] != -1:  
            
            # Use NumPy to find indices of all the childs of this cnt
            child_indices = np.where(hierarchy[:, 3] == i)[0]
            child_area = 0
            for idx in child_indices : child_area += cv2.contourArea(contours[idx])
            
            if cv2.contourArea(contours[i]) > 1.5*child_area: 
                ambiguous_parent_cnts.append(contours[i])
                
            else: parent_cnts.append(contours[i])
    
            to_del.append(i)
            
        if hierarchy[i][3] != -1:  
            child_cnts.append(contours[i])
            to_del.append(i)
        
    orphan_cnts = contours.copy()
    for i in reversed(sorted(to_del)): del orphan_cnts[i] # del contours ID as parent of child
    
    return parent_cnts, child_cnts, orphan_cnts, ambiguous_parent_cnts


def dilate(image, size):
    
    # Définir le noyau de dilatation
    kernel = np.ones((size, size), np.uint8)  # Vous pouvez ajuster la taille du noyau selon vos besoins
    
    # Dilater les pixels dans l'image
    dilated_image = cv2.dilate(image, kernel, iterations = 1)
    
    return dilated_image

    
def find_closest_points(contour1, contour2):
    
    # Convert contours into lists of point coordinates
    contour1_points = contour1.squeeze().tolist()
    contour2_points = contour2.squeeze().tolist()

    # Build KD-Trees for both contours
    tree2 = KDTree(contour2_points)

    # Query KD-Tree for closest points
    distances, indices = tree2.query(contour1_points)

    # Handle multiple pairs with the same minimum distance
    min_distance = min(distances)
    best_pairs = [(p1, contour2_points[indices[i]]) for i, p1 in enumerate(contour1_points) if distances[i] == min_distance]
    
    best_pairs = best_pairs[0]

    return best_pairs


def erases_cnts_under_area_thresh (binarized_pruned, area_threshold):
    ''' Based on a binary image,
    This function pain in black cnts whose area is under a given threshold'''
    cnts, _ = retrieve_contours(binarized_pruned, 'external')
    if len(cnts) > 1200 :
        return binarized_pruned
    
    for cnt in cnts:
        if cv2.contourArea(cnt) < area_threshold:
            cv2.drawContours(binarized_pruned, [cnt], -1, 0, -1)     # erase them
            
    return binarized_pruned


def distance_between_cnts(contour1, contour2):
    
    # Approximation de polygone pour réduire le nombre de points dans les contours
    epsilon = 0.02 * cv2.arcLength(contour1, True)  # Vous pouvez ajuster ce paramètre selon vos besoins
    approx_contour1 = cv2.approxPolyDP(contour1, epsilon, True)
    approx_contour2 = cv2.approxPolyDP(contour2, epsilon, True)
    
    # Calcul de la distance euclidienne entre les points des contours approximés
    min_distance = float('inf')  # Initialisez la distance minimale avec une valeur infinie
    
    for point1 in approx_contour1:
        for point2 in approx_contour2:
            distance = np.linalg.norm(np.array(point1[0]) - np.array(point2[0]))
            if distance < min_distance:
                min_distance = distance
    
    return min_distance


#### main steps function

def ID_obvious_childs_and_parents (binarized, max_cnts):
    
    '''This function ID obvious parent and child on a binarized image'''

    cnt = None
    # erase ok cnts on a copy
    binarized_pruned = binarized.copy()
   
    # creat the final negatif where put detected cnts
    final_negatif = np.zeros(binarized.shape, dtype=np.uint8)

    # retrieve cnts
    contours, hierarchy = retrieve_contours(binarized, 'all')
    
    if len(contours)<1 or len(contours)>max_cnts: 
        return binarized_pruned, final_negatif, cnt
    
    # classify cnts
    parent_cnts, child_cnts, orphan_cnts, ambiguous_parent_cnts = classify_cnts(contours, hierarchy)
    
    cnt = [contours, hierarchy, parent_cnts, child_cnts, orphan_cnts, ambiguous_parent_cnts]
        
    # draw parrent and childs
    cv2.drawContours(final_negatif, child_cnts, -1, 255, -1)
    cv2.drawContours(final_negatif, parent_cnts, -1, 255, -1)
    
    cv2.drawContours(binarized_pruned, parent_cnts, -1, 0, -1)
    
    return binarized_pruned, final_negatif, cnt


def find_new_children_by_dilatation (binarized_pruned, final_negatif,
                                     contours, hierarchy,
                                     dilatation_power, min_area_considered_child, min_dist_child_parent):


    
        # dilate image
        dilated_img = dilate(binarized_pruned, dilatation_power)
        
        # retrieve cnt in it
        contours_dil, hierarchy_dil = retrieve_contours(dilated_img, 'all')
        
        # classify these cnts
        parent_cnts_dil, child_cnts_dil, orphan_cnts_dil, ambiguous_parent_cnts_dil = classify_cnts(contours_dil, hierarchy_dil)
        
        # remove too small cnts
        child_cnts_dil = [cnt for cnt in child_cnts_dil if cv2.contourArea(cnt) > min_area_considered_child]
        
        # correct_previous parent dilatation by dilating the children
        black_image = np.zeros(binarized_pruned.shape, dtype=np.uint8)
        cv2.drawContours(black_image, child_cnts_dil, -1, 255, -1) 
        black_image = dilate(black_image, dilatation_power)
        child_cnts_dil, _ = retrieve_contours(black_image, 'external')
    
        # we want to know who is the original parent of the child cnt finded
        ID_parents = []
        for child in child_cnts_dil : 
            child_area = cv2.contourArea(child) 
            for i, parent in enumerate (contours):   
                if hierarchy[i][3] == -1 :
                    # test if the child is at leat partialy in the potential parents
                    if cnt1_partly_in_cnt2(child, cv2.convexHull(parent), min_dist_child_parent): ID_parents.append(i)
                
        # keep unique values in parents
        ID_parents = list(set(ID_parents))    
        
        # remove ID parents from the original binarized_pruned and add them and the final_negatif with their childs
        cv2.drawContours(final_negatif, child_cnts_dil, -1, 255, -1)
        for i in ID_parents : 
            cv2.drawContours(binarized_pruned, [contours[i]], -1, 0, -1)     # erase them
            cv2.drawContours(final_negatif, [contours[i]], -1, 255, -1)      # draw them
        
        return binarized_pruned, final_negatif


def link_neibors_cnts(final_negatif, thresh_dist, line_thickness):
    '''This function draw link cnts by their closest point, at the condition the gap is under a threshold'''

    # retrieve cnts
    contours_dist, hierarchy_dist = retrieve_contours(final_negatif, 'all')
    if len(contours_dist)<1:
        return final_negatif
    

    # test if the mifdle of the cnt is blank
    cnt_to_test = []
    for i, cnt in enumerate(contours_dist):
        
        if cv2.contourArea(cnt) > 40:
            centro = centroid_calculator(cnt) 
            if final_negatif[centro[1], centro[0]] == 0 : 
                cnt_to_test.append(i)
            
    if len(cnt_to_test) > 0 :          
        # try to link these cnt with others
        tested = []             
        for i in cnt_to_test:
            
            for j, cnt2 in enumerate(contours_dist):
                if i != j and (i,j) not in tested and len(cnt2) > 1: 
                    cnt1 = contours_dist[i]
                    distance = distance_between_cnts(cnt1, cnt2)
                    if distance < thresh_dist :
                        
                        #record them to avoid multiple test
                        tested.append((i,j)); tested.append((j,i))
                        
                        #find closest points between the cnts 
                        closest_point1, closest_point2 = find_closest_points(cnt1, cnt2)
                        
                        # draw a line between these cnt
                        cv2.line(final_negatif, closest_point1, closest_point2, 255, line_thickness)
                        
    return final_negatif
  

#### retrieve cuticle among detected cnts
def filter_segmented_image(mask, merge_contour):  
    
    # This function retrieve the contours inside the color segmented image and filter the contours according the following rules:
        # Contours inside another contour should correspond to their direct internal border (i.e. child)
        # The perimeter of the contour is up to 40 pixels
        
    # usefull link:
        # Contour detection and hierarchy: https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    
    
    # Step 1 - find contours and their hierarchy
    ############################################
        
    all_contours, hierarchy_all = retrieve_contours(mask, 'all')
    
    if not len(all_contours) > 0 : # verify that contours were detected
        return np.zeros(mask.shape).astype(mask.dtype)
        # if no cnt was found, return a black image at the dimension of the orignal image

    
    ext_contours, hierarchy_ext = retrieve_contours(mask, 'external')

    # Step 2 - Filter external contours: remove the very small ones
    #####################################
            
    ext_contours = [contour for contour in ext_contours if cv2.arcLength(contour, True) >= 40]
    
    
    
    # Step 3 - Filter internal contours: keep only cnt who are child of an external cnt
    ###################################
            
    # Firstly, Identify index of external cnt in the all_contour list
    true_ext_cnt = [] # will contain the index
    for i in range(len(all_contours)):
        current_hierarchy = hierarchy_all[i]
        current_contour = all_contours[i]
        if current_hierarchy[3] == [-1] and not cv2.arcLength(current_contour,True) < 40 : 
            
            for ext_cnt in ext_contours : # try to match the current contour with the real list of external contours
                if cv2.matchShapes(current_contour, ext_cnt, cv2.CONTOURS_MATCH_I1, 0) == 0 : true_ext_cnt.append(i); break
    
    
    # Then, Filter in the list of all cnt to keep the child
    int_contours=[]
    for i in range(len(all_contours)):
        
        current_hierarchy = hierarchy_all[i]
        current_contour = all_contours[i]
        
        if current_hierarchy[3] in true_ext_cnt and not cv2.arcLength(current_contour,True) < 40 : 
            int_contours.append(current_contour)


    # Step 3 - Draw the final image
    ###############################
    contours = ext_contours + int_contours
    result = draw_cnts(mask, contours)
    
    if len(ext_contours) > len(int_contours) and len(ext_contours) < 500 : result = filter_segmented_image2(result, merge_contour) # If cuticle is not close, start a second filter using convex hull 
            
    return result 


def filter_segmented_image2(mask, merge_contour):  
    
    ## Contour filtering II: for the case when cuticle is not close, external cnt inside other are removed using convex hull ##
    ###########################################################################################################################
    
    # Step 1 - find contours and their hierarchy
    ############################################

    all_contours, hierarchy_all = retrieve_contours(mask, 'all')
    ext_contours, hierarchy_ext = retrieve_contours(mask, 'external')
    
    ext_contours = [contour for contour in ext_contours if cv2.arcLength(contour, True) >= 40]


    # Step 2 - Filtre external cnt that are inside another external one
    ###################################################################

    to_supr_ext = []
    for i in range (len(all_contours)):
        current_contour = all_contours[i]
        current_hierarchy = hierarchy_all[i]
        
        if current_hierarchy[3] == -1 : # identify external cnt
            
            for ext_cnt in ext_contours :
                
                if not cv2.matchShapes(current_contour, ext_cnt, cv2.CONTOURS_MATCH_I1, 0) == 0 : # avoid to compare the cnt with imself
                
                    hull = cv2.convexHull(ext_cnt) # draw a convex hull arrond the cnt
                    
                    if cnt1_inside_cnt2(current_contour, hull, 10) : to_supr_ext.append(i); break # if curent cnt is inside the bounding box
            
    # Step 3 - Filtre internal cnt that are child cnt of extenral cnt identified as irrelevant
    ##########################################################################################
    
    to_supr_int = [i for i, current_hierarchy in enumerate(hierarchy_all) if current_hierarchy[3] in to_supr_ext]


    # Remove irrelevant cnts from the list
    ######################################
    tospp = to_supr_ext + to_supr_int # merge the two list
    for i in reversed(sorted(tospp)): 
        del all_contours[i]  # del contours identified as irrelevant
      

    # Step 3 - Draw the final image
    ###############################
    result = draw_cnts(mask, all_contours)
    
    if merge_contour == True and len(all_contours) > 0 : result = filter_segmented_image3(result)

    return result



def filter_segmented_image3(mask):  
    
    ## Contour filtering III: for the case when cuticle is very thin, leading to missing cnts. Solytary cnts respecting the rules are merged in one using convex hull ##
    ####################################################################################################################################################################
    
    # Rely on the following rules:
         # Contours included in the convex hull should appear as single (i.e. without parents and childs) both before and after pixel dilatation 
         # The convex hull should not cross any children contours
    
    initial_cnt, initial_hierarchy = retrieve_contours(mask, 'all')

    # Step 1 - Identify cnt that are single despite pixel dilatation
    ################################################################
        
    # modify image I: close cnt
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, rect_kernel)

    # modify image II: apply morphological gradiant if they can  contain a cnt
    kernel = np.ones((6,6),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    
    # find all contours on the modified image
    modified_contours, hierarchy_modified = retrieve_contours(mask, 'all')

    
    # identify single contours on the modified image
    modified_single = []
    for i in range(len(modified_contours)):
        current_hierarchy = hierarchy_modified[i]
        if current_hierarchy[3] == -1 and current_hierarchy[2] == -1 : # identify single cnt
            
            if black_insid_hull(mask, modified_contours[i]) : # identify cnt that are not simple dot (eg. legs)
                modified_single.append(modified_contours[i]) 
    

    # Step 2 - From the initial list, identify cnt that are single and match the list of modified singles
    #####################################################################################################
    
    single = []
    for i in range(len(initial_cnt)) : 
        current_cnt = initial_cnt[i]
        current_hierarchy = initial_hierarchy[i]
        
        # find cnt that are single
        if current_hierarchy[3] == -1 and current_hierarchy[2] == -1 :  
                
            # try to match it with the initial list of single cnt
            for modified_cnt in modified_single :
                if  cnt1_inside_cnt2(current_cnt, modified_cnt, -10) : single.append(current_cnt); break
            
    # Step 3 - Creat the convex hull and remove contours inside
    ###########################################################
    
    if len(single) > 0 :
        
        # Creat the convex hull
        #######################
        
        contours = np.vstack(single) # merge the contours in one 
        ctr = cv2.convexHull(contours) # create a convex hull around this contour
        
        # Filtre external cnt that are inside the convex hull
        #####################################################
    
        to_supr = []
        for i in range (len(initial_cnt)):
            
            current_contour = initial_cnt[i]
            current_hierarchy = initial_hierarchy[i]
            
            if current_hierarchy[3] == -1 and current_hierarchy[2] == -1 : 
                
                if cnt1_inside_cnt2(current_contour, ctr, 10) : to_supr.append(i);  # if curent cnt is inside the bounding box
                
        # Remove cnts from the list
        for i in reversed(sorted(to_supr)): del initial_cnt[i] # del contours identified as irrelevant

        
    result = draw_cnts(mask, initial_cnt)
    
    if len(single) > 0 : cv2.drawContours(result, [ctr], -1, (255, 255, 255), 1)  #draw contour in blank
        
    return result
  

#### final function

def fill_contours(binarized, max_cnts = 150):
    
    if np.sum(binarized) == 0:
        return binarized
        
    ###############################################☻####
    ####### STEP 1: ID obvious childs and parents ######
    ####################################################    
  
    binarized_pruned, final_negatif, cnts = ID_obvious_childs_and_parents (binarized, max_cnts)

    if cnts == None :
        return final_negatif
    
    contours = cnts[0]
    hierarchy = cnts[1]
    parent_cnts = cnts[2]
    child_cnts= cnts[3]
    orphan_cnts = cnts[4]
    ambiguous_parent_cnts = cnts[5]
    
    # display((binarized, binarized_pruned, final_negatif) ,0, 20)  # end of the step 1
            
    ###############################################☻######
    ####### STEP 2:  try a first slyght dilatation ######
    ###############################################☻######
    # if len(orphan_cnts) > 0 or len(ambiguous_parent_cnts) : 
    if not np.sum(binarized_pruned) == 0 :
 
        min_area_considered_child = 1000 # in px
        dilatation_power = 10
        min_dist_child_parent = -1 #negative for outside
        
        binarized_pruned, final_negatif = find_new_children_by_dilatation (binarized_pruned, final_negatif,
                                             contours, hierarchy,
                                             dilatation_power, min_area_considered_child, min_dist_child_parent)
        # end of the step 2
        # display((binarized, binarized_pruned, final_negatif) ,0, 20)
                                  
    ###############################################☻######
    ####### STEP 3:  try a massive dilatation ######
    ###############################################☻######
    # if len(ID_parents) < (len(orphan_cnts) + len(ambiguous_parent_cnts)):
    if not np.sum(binarized_pruned) == 0 :

        dilatation_power = 50
        binarized_pruned, final_negatif = find_new_children_by_dilatation (binarized_pruned, final_negatif,
                                              contours, hierarchy,
                                              dilatation_power, min_area_considered_child, min_dist_child_parent)
    # end of the step 3
    #display((binarized, binarized_pruned, final_negatif) ,0, 20)


        ###############################################☻######
        ####### STEP 4:  refine cnt and put all cnt together ######
        ###############################################☻######
    
    if not np.sum(binarized_pruned) == 0 :
    
        # put all cnt together 
        cnt_fin, hirarchy_fin = retrieve_contours(binarized_pruned, 'all')
        cv2.drawContours(final_negatif, cnt_fin, -1, 255, -1) 
            
        # fill little gap
        cnt_fin, hirarchy_fin = retrieve_contours(final_negatif, 'all')   
        _, child_cnts_fin, _, _ = classify_cnts(cnt_fin, hirarchy_fin)
        cv2.drawContours(final_negatif, child_cnts_fin, -1, 255, -1)      # draw them
            
        # end of the step 4
        # display((binarized, binarized_pruned, final_negatif) ,0, 20)
        
        ###############################################☻#############
        ####### STEP 5:  try to link remaining dist with lines ######
        ###############################################☻#############

        thresh_dist = 50   
        line_thickness = 5
        
        final_negatif = link_neibors_cnts(final_negatif, thresh_dist, line_thickness)
      
        ###############################################☻############
        ####### STEP 6:  refine cnt and fill inatributed ones ######
        ###############################################☻############
    
        # fill new childs
        cnt_fin, hirarchy_fin = retrieve_contours(final_negatif, 'all')
               
        _, child_cnts_fin, orphan, ambiguous = classify_cnts(cnt_fin, hirarchy_fin)
        cv2.drawContours(final_negatif, child_cnts_fin, -1, 255, -1)      # draw child

        # hande the case of last orphan and ambiguous
        for cnt in orphan + ambiguous:
            
            # first we filter them, to keep only the actual empty ones
            black_image = np.zeros(binarized.shape, dtype=np.uint8)
            cv2.drawContours(black_image, [cnt], -1, 255, -1) 
            
            # a fully white cnt would give 1. 0.5 mean the cnt contain a lot of empty part
            if cv2.contourArea(cv2.convexHull(cnt)) == 0 or np.sum(black_image)/ 255 / cv2.contourArea( cv2.convexHull(cnt) ) < 0.5 :
                # draw selected cnt with an convex hull
                cv2.drawContours(final_negatif, [cv2.convexHull(cnt)], -1, 255, -1)      # draw them
            
    
    return final_negatif



def segmente_scan(environment, segmentation_parameters, outputs):
    
    environment, segmentation_parameters, outputs = input_cheking(environment, segmentation_parameters, outputs)
    
    # Unfold outputs
    input_folder, output_folder, scan_label = environment
    threshold, merge_contour, vial_effort, search_spec_limits = segmentation_parameters
    seg_image_gap, xslx_record, image_record, show_images = outputs

        
    if not xslx_record and not image_record and not show_images:
        raise ValueError("No output requested. \nSetup either show_images, image_record or xslx_record.")
    
        
    # Creat necessary directories 
    #############################
    
    if xslx_record or image_record:
    
        output_file = os.path.join(output_folder, scan_label)
        if not os.path.exists(output_file): os.makedirs(output_file)
        else: print('\n\nCaution: a folder already exist for this scan. Old data might be erased or mixed with new data\n\n')
            
    if xslx_record:
        
        # Creat the xlsx file
        xfile_name = os.path.join(output_file, f"{scan_label}.xlsx")
        wb = Workbook()
        wb.save(filename = xfile_name)
        
        # Open the xlsfile
        xfile = openpyxl.load_workbook(xfile_name) # open the xfile
        sheet = xfile[xfile.sheetnames[0]] # Attribute the first sheet as the place of interest
        
    
    # Loads the image from the input folder 
    #######################################
    path_3D_scan = os.path.join(input_folder, scan_label, 'slices-blend-8-bit.tif')
    
    # Pre-analyse the sample  
    ########################
    
    if search_spec_limits :
        # Detect the begening and end of the invidual within the scan
        beginning, end = beginning_and_end(path_3D_scan, segmentation_parameters) 
    
    else :
        beginning = 0
        end = count_2d_images(path_3D_scan)
        
    # Start image segmentation  
    ##########################
    
    for img_nb in tqdm (range(beginning, end, seg_image_gap), desc="Segmente images…", ascii=False, ncols=75): 
                
        image = open_single_img_tif(path_3D_scan, img_nb)
        
        # Process image colors
        if not vial_effort : adjusted_image = color_image_processing(image) 
        else : adjusted_image = color_image_processing_2(image) 
        
        # Remove the vial contours if finded
        without_vial = remove_vial(adjusted_image) 

        # Segmmente image according the threshold
        segmented = segmente_image(without_vial, threshold) 
        
        # Apply a second function to ensure that the vial is removed
        if vial_effort : segmented = remove_vial_2(segmented, vial_effort) 
        
        # Keep only the most external contours according contours hierarchy
        segmented_filtered = filter_segmented_image(segmented, merge_contour) 
        
        # Fill every contours to extract the body volume
        negatif = fill_contours(segmented_filtered) 
                  
        # Record result in the xfile
        ############################ 
        if xslx_record: 
            
            #Extract the body volume and the cuticle volume from images
            body_volume = get_metrics(negatif) 
            cuticle_volume = get_metrics(segmented_filtered) 
                  
            row = img_nb + 1 # get the row to write in the xlsx file
            
            # Write body volume
            place_data = f'{chr(ord("@") + 1)}{row}'  # Build the place in the xfile where to write the data
            sheet[place_data] = body_volume  # Write the data in the specified place
            
            # Write cuticle volume
            place_data = f'{chr(ord("@") + 2)}{row}'  # Build the place in the xfile where to write the data
            sheet[place_data] = cuticle_volume  # Write the data in the specified place


        # Save the segmented image
        ############################
        
        if image_record :
            if img_nb % image_record == 0 : 
                                                
                number = str(img_nb).zfill(4)
                img_name = f'segmented_{scan_label}_{number}.tif'
                image_outupt_path = os.path.join(output_file, img_name)
                
                # concatenate the image used to extract the cuticle volume and the one for the body volume
                numpy_vertical_concat = np.concatenate((segmented_filtered, negatif), axis=1) 
                cv2.imwrite(image_outupt_path, numpy_vertical_concat) # save the image in the specified output folder
                        
         # visualize the image
        if show_images : 
            if img_nb % show_images[0] == 0 :
                numpy_vertical_concat = np.concatenate((image, segmented_filtered, negatif), axis=1)
                display(numpy_vertical_concat, show_images[1], show_images[2], 'original slice - segmentation cuticle - segmentation body')
        
             
    if xslx_record: xfile.save(xfile_name) # Save updated xfile
 

#### control inputs
def input_cheking(environment, segmentation_parameters, outputs):
    
    # Unfold outputs
    scan_path, output_folder, scan_label = environment
    threshold, merge_contour, vial_effort, search_spec_limits = segmentation_parameters
    seg_image_gap, xslx_record, image_record, show_images = outputs 
    
    
    if not xslx_record and not image_record and not show_images:
        raise ValueError("No output requested. \nSetup either show_images, image_record or xslx_record.")
    
    
    ##### Verification of the environment parameters
    
    if not os.path.exists(scan_path): 
        raise ValueError("Input folder doesn't exist.")
        
    if not scan_label:
        path_components = scan_path.split(os.sep) # Split the path into its components
        scan_label = path_components[-2] # Get the folder name containing the specimen label assuming is position

    # Verification of the segmentation parameters
    
    if not isinstance(search_spec_limits, bool):
        raise ValueError("Please set 'search_spec_limits' as True or False according to your need to crop empty part of the scan")
     
    if not isinstance(threshold, list) or len(threshold) != 2 or not all(isinstance(val, (int, float)) and 0 <= val <= 255 for val in threshold):
        raise ValueError("Please set 'threshold' as two figures comprised between 0 and 255.\ne.g., [128, 255]")
    
    if not isinstance(merge_contour, bool):
        raise ValueError("Please set 'merge_contour' as True or False according to your need to merge contour in a single one")
    
    if not (isinstance(vial_effort, int) and vial_effort > 0) and not (isinstance(vial_effort, float) and 0 < vial_effort < 1) and vial_effort is not False or vial_effort is True:
        raise ValueError("Please set 'vial_effort' as False to not use this function or as the expected minimum expected ratio between vial area / image area (e.g., 0.25)")
    
    
    ##### Verification of the output parameters
    
    if not isinstance(xslx_record, bool):
        raise ValueError("Please set 'xslx_record' as True or False according to your need to save result in Excel")
    
    if image_record == True: image_record = 1
    elif image_record == 0: image_record = False
    elif not isinstance(image_record, int) or image_record < 0 :
        raise ValueError("Please set 'image_record' as \n 0 or False for no image saved \n 1 or True for all images saved \n 2 for one image out of two \n 3 for one image out of three etc")
    
    if not isinstance(seg_image_gap, int) or seg_image_gap < 1:
        raise ValueError("Please set 'seg_image_gap' as a number greater than or equal to 1.\n 1 to segment all images \n 2 to segment one image out of two \n 3 to segment one image out of three \n etc")
    
    if not isinstance(show_images, list) and show_images is not False:
        raise ValueError("Please set 'show_images' either as False or as following: \n[frequency, swhowing time in mls, rescale factor]\ne.g., [20, 0, 20]")
    
    # Fold inputs
    environment = [scan_path, output_folder, scan_label]
    segmentation_parameters = [threshold, merge_contour, vial_effort, search_spec_limits]
    outputs = [seg_image_gap, xslx_record, image_record, show_images]
    
    return environment, segmentation_parameters, outputs



def auto_exoskeleton_segmentation(environment, segmentation_parameters, outputs):
    
    environment, segmentation_parameters, outputs = input_cheking(environment, segmentation_parameters, outputs)
    
    # Unfold outputs
    scan_path, output_folder, scan_label = environment
    threshold, merge_contour, vial_effort, search_spec_limits = segmentation_parameters
    seg_image_gap, xslx_record, image_record, show_images = outputs

        
    # Creat necessary directories 
    #############################
    
    if xslx_record or image_record:
    
        output_file = os.path.join(output_folder, scan_label)
        if not os.path.exists(output_file): os.makedirs(output_file)
        else: print('\n\nCaution: a folder already exist for this scan. Old data might be erased or mixed with new data\n\n')
            
    if xslx_record:
        
        # Creat the xlsx file
        xfile_name = os.path.join(output_file, f"{scan_label}.xlsx")
        wb = Workbook()
        wb.save(filename = xfile_name)
        
        # Open the xlsfile
        xfile = openpyxl.load_workbook(xfile_name) # open the xfile
        sheet = xfile[xfile.sheetnames[0]] # Attribute the first sheet as the place of interest
                
    # Pre-analyse the sample  
    ########################
    
    if search_spec_limits :
        # Detect the begening and end of the invidual within the scan
        beginning, end = beginning_and_end(scan_path, segmentation_parameters) 
    
    else :
        beginning = 0
        end = count_2d_images(scan_path)
        
    # Start image segmentation
    ##########################
    
    for img_nb in tqdm (range(beginning, end, seg_image_gap), desc="Segmente images…", ascii=False, ncols=75): 
                
        image = open_single_img_tif(scan_path, img_nb)
        
        # Process image colors
        if not vial_effort : adjusted_image = color_image_processing(image) 
        else : adjusted_image = color_image_processing_2(image) 
        
        # Remove the vial contours if finded
        without_vial = remove_vial(adjusted_image) 

        # Segmmente image according the threshold
        segmented = segmente_image(without_vial, threshold) 
        
        # Apply a second function to ensure that the vial is removed
        if vial_effort : segmented = remove_vial_2(segmented, vial_effort) 
        
        # Keep only the most external contours according contours hierarchy
        segmented_filtered = filter_segmented_image(segmented, merge_contour) 
        
        # Fill every contours to extract the body volume
        negatif = fill_contours(segmented_filtered) 
                  
        # Record result in the xfile
        ############################ 
        if xslx_record: 
            
            #Extract the body volume and the cuticle volume from images
            body_volume = get_metrics(negatif) 
            cuticle_volume = get_metrics(segmented_filtered) 
                  
            row = img_nb + 1 # get the row to write in the xlsx file
            
            # Write body volume
            place_data = f'{chr(ord("@") + 1)}{row}'  # Build the place in the xfile where to write the data
            sheet[place_data] = body_volume  # Write the data in the specified place
            
            # Write cuticle volume
            place_data = f'{chr(ord("@") + 2)}{row}'  # Build the place in the xfile where to write the data
            sheet[place_data] = cuticle_volume  # Write the data in the specified place

        # Save the segmented image
        ############################
        
        if image_record :
            if img_nb % image_record == 0 : 
                                                
                number = str(img_nb).zfill(4)
                img_name = f'segmented_{scan_label}_{number}.tif'
                image_outupt_path = os.path.join(output_file, img_name)
                
                # concatenate the image used to extract the cuticle volume and the one for the body volume
                numpy_vertical_concat = np.concatenate((segmented_filtered, negatif), axis=1) 
                cv2.imwrite(image_outupt_path, numpy_vertical_concat) # save the image in the specified output folder
                        
         # visualize the image
        if show_images : 
            if img_nb % show_images[0] == 0 :
                numpy_vertical_concat = np.concatenate((image, segmented_filtered, negatif), axis=1)
                display(numpy_vertical_concat, show_images[1], show_images[2], 'original slice - segmentation cuticle - segmentation body')
        
             
    if xslx_record: xfile.save(xfile_name) # Save updated xfile
 