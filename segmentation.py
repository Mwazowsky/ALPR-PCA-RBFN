import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from skimage.feature import canny
from skimage import filters

from PIL import Image

from scipy.stats import mode
from skimage import img_as_ubyte
import scipy
import warnings
import os
import time

warnings.filterwarnings("ignore")

class NumberPlateSegmenter:
    def __init__(self, img_path):
        self.img_path = img_path
    
    def hough_transform(self, input_img):
        image_gscl = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) 
        img_filter = filters.median(image_gscl, np.ones((3, 3)))
        edges = canny(img_filter)
        # Classic straight-line Hough transform between 0.1 - 180 degrees.
        tested_angles = np.deg2rad(np.arange(0.1, 180.0))
        h, theta, d = hough_line(edges, theta=tested_angles)
        accum, angles, dists = hough_line_peaks(h, theta, d)
        most_common_angle = mode(np.around(angles, decimals=2))[0]
        skew_angle = np.rad2deg(most_common_angle - np.pi/2)
        img_rotated = rotate(img_filter, skew_angle[0], cval=0)

        return img_rotated
    
    def verifySizes(self, contour):
        area = cv2.contourArea(contour)
        return area > 150
    
    def find_contours(self, ori_img, img):
        # Find all contours in the image
        cntrs, _hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = _hierarchy[0]  # get the actual inner list of hierarchy descriptions

        # Grab only the innermost child components
        cntrs = [c[0] for c in zip(cntrs, hierarchy) if c[1][3] == -1]

        # Check largest 5 or  15 contours for license plate or character respectively
        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

        x_cntr_list = []
        img_res = []
        c_count = 0
        for i, cntr in enumerate(cntrs):
            # detects contour in binary image and returns the coordinates of rectangle enclosing it
            intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

            # checking the dimensions of the contour to filter out the characters by contour's size
            if(self.verifySizes(cntr)):
                x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours
                char = ori_img[intY:intY+intHeight, intX:intX+intWidth]
                img_res.append(char) # List that stores the character's binary image (unsorted)
        indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
        img_res_copy = []
        for idx in indices:
            img_res_copy.append(img_res[idx])# stores character images according to their index
        img_res = img_res_copy

        return img_res
    
    def segment_word(self, image, bin_image):
        word_list = self.find_contours(image, bin_image)

        return word_list
    
    def character_segmentaion(self, word_bin_img_):
        ith_word_erode = scipy.ndimage.morphology.binary_erosion(word_bin_img_,  structure=np.ones((1,5))).astype(word_bin_img_.dtype)
        ith_word_dilate = scipy.ndimage.morphology.binary_dilation(ith_word_erode,  structure=np.ones((10,1))).astype(ith_word_erode.dtype)

        ith_word_dilate = cv2.subtract(255, ith_word_dilate)

        vertical_projection = np.sum(ith_word_dilate, axis=0)

        height_avg = np.max(vertical_projection)

        whitespace_lengths = []
        whitespace = 0
        index = 0
        index_=[]
        for vp in vertical_projection:
            if vp >= height_avg:
                whitespace = whitespace + 1
            elif vp <= height_avg:
                if whitespace != 0:
                    whitespace_lengths.append(whitespace)
                    index_.append(index)
                whitespace = 0 # reset whitepsace counter. 
            index+=1
        avg_white_space_length = np.mean(whitespace_lengths) + 5

        whitespace_length = 0
        divider_indexes = []
        for index, vp in enumerate(vertical_projection):
            if vp >= height_avg:
                whitespace_length = whitespace_length + 1
            elif vp <= height_avg:
                if whitespace_length != 0 or whitespace_length > (avg_white_space_length):#change here
                    divider_indexes.append(index-int(whitespace_length/2))
                    whitespace_length = 0 # reset it

        for index, vp in reversed(list(enumerate(vertical_projection))):# for last word segmentation
            if vp >= height_avg:
                whitespace_length = whitespace_length + 1
            elif vp <= height_avg:
                if whitespace_length != 0 or whitespace_length > (avg_white_space_length):
                    divider_indexes.append(index+int(whitespace_length/2))
                    break
                    whitespace_length = 0 # reset it

        divider_indexes = np.array(divider_indexes)
        dividers = np.column_stack((divider_indexes[:-1],divider_indexes[1:]))

        img_chr = []
        for i, window in enumerate(dividers):
            ith_chr = word_bin_img_[:,window[0]:window[1]]
            if ith_chr.shape[1] >= 15:
                char_img = Image.fromarray(ith_chr)
                resized_char_image = char_img.resize((20, 30), resample=Image.BILINEAR)
                resized_char_matrix = np.array(resized_char_image)
                img_chr.append(resized_char_matrix)

        return img_chr
    
    def segmentation(self, rotated_img_):
        cv_img = img_as_ubyte(rotated_img_)
        img_gray_lp = cv2.resize(cv_img, (333, 95))

        is_light = np.mean(img_gray_lp) < 80
        if is_light:
            # define the alpha and beta
            alpha = 10                  # Contrast control
            beta = 5                    # Brightness control
            img_gray_lp = cv2.convertScaleAbs(img_gray_lp, alpha, beta)

        _, img_binary_lp = cv2.threshold(img_gray_lp, 190, 255, cv2.THRESH_BINARY)

        unique, counts = np.unique(img_binary_lp, return_counts=True)
        pixel_cnt = dict(zip(unique, counts))

        if pixel_cnt[0] < pixel_cnt[255]:
            img_binary_lp = (255 - img_binary_lp)

        img_binar = img_binary_lp

        img_binary_lp = scipy.ndimage.morphology.binary_erosion(img_binary_lp,  structure=np.ones((6,1))).astype(img_binary_lp.dtype)

        (rows,cols) = img_binary_lp.shape
        v_projection = np.array([x / cols for x in img_binary_lp.sum(axis=1)])
        v_threshold = np.mean(v_projection)
        v_black_areas = np.where(v_projection < v_threshold)
        img_v_binary_lp = img_binary_lp.copy()

        for j in v_black_areas:
            img_v_binary_lp[j, :] = 0
        v_proj_new = np.array([x / cols for x in img_v_binary_lp.sum(axis=1)])

        line_start = 0
        line_end = 0

        for c, x in enumerate(v_proj_new[:60]):
            if x == 0:
                if c <= 20:
                    line_start = c - 7
                line_end = c

        cropped_image_bin = img_binar[line_start:line_end, 0:]
        cropped_image_bin = scipy.ndimage.morphology.binary_dilation(cropped_image_bin,  structure=np.ones((5,3))).astype(cropped_image_bin.dtype)
        cropped_image_bin = scipy.ndimage.morphology.binary_erosion(cropped_image_bin,  structure=np.ones((3,3))).astype(cropped_image_bin.dtype)

        ### Debris Cleaning With CCA
        analysis = cv2.connectedComponentsWithStats(cropped_image_bin, 4, cv2.CV_32S)
        (totalLabels, label_ids, values, centroid) = analysis

        output = np.zeros(cropped_image_bin.shape, dtype="uint8")
        for i in range(1, totalLabels):
            area = values[i, cv2.CC_STAT_AREA]
            width = values[i, cv2.CC_STAT_WIDTH]
            if area > 205 and width < 40 and width > 10:
                componentMask = (label_ids == i).astype("uint8") * 255
                output = cv2.bitwise_or(output, componentMask)

        ### Word Level Segmentation
        wrd_dilated = scipy.ndimage.morphology.binary_dilation(output,  structure=np.ones((3,20))).astype(output.dtype)
        img_wrd = self.segment_word(output, wrd_dilated)

        ### Character Level Segmentation    
        img_chrs = []
        for i, word in enumerate(img_wrd):
            ith_char = self.character_segmentaion(word)
            img_chrs.append(ith_char)

        return np.asarray(img_chrs)
    
    def segment_word(self, image, bin_image):
        word_list = self.find_contours(image, bin_image)
        return word_list
    
    def find_contours(self, ori_img, img):
        cntrs, _hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = _hierarchy[0]
        cntrs = [c[0] for c in zip(cntrs, hierarchy) if c[1][3] == -1]
        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
        x_cntr_list = []
        img_res = []
        c_count = 0
        for i, cntr in enumerate(cntrs):
            intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
            if self.verify_sizes(cntr):
                x_cntr_list.append(intX)
                char = ori_img[intY:intY+intHeight, intX:intX+intWidth]
                img_res.append(char)
        indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
        img_res_copy = []
        for idx in indices:
            img_res_copy.append(img_res[idx])
        img_res = img_res_copy
        return img_res
    
    def verify_sizes(self, contour):
        area = cv2.contourArea(contour)
        return area > 150

    def segment_plate(self):
        start_time = time.time() 
        
        img = cv2.imread(self.img_path)

        rotated_img = self.hough_transform(img)
        segmented_chars = self.segmentation(rotated_img)

        temp_dir = "./temp/"
        word_folder = os.path.join(temp_dir, 'word_0')
        os.makedirs(word_folder, exist_ok=True)

        word_count = len(segmented_chars)
        max_char_count = max(len(word) for word in segmented_chars)
        char_dimension = segmented_chars[0][0].shape

        char_array = np.zeros((word_count, max_char_count, char_dimension[0], char_dimension[1]), dtype=np.uint8)
        
        base_name = os.path.splitext(os.path.basename(self.img_path))[0]
        last_j = 0

        for i, word in enumerate(segmented_chars):
            for j, char in enumerate(word):
                # Calculate the sequential index based on i, j, and last_j
                sequential_index = i + j + last_j
                # Create the new file name using the base name and sequential index
                new_file_name = f"{sequential_index}_{base_name[sequential_index]}.jpg"

                save_path = os.path.join(word_folder, new_file_name)
                cv2.imwrite(save_path, char)
                char_array[i, j] = char  # Add the character to the char_array

            # Update the last_j value
            last_j += len(word)

        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time
        # print(f"Segmentation took {execution_time} seconds")

        return char_array
