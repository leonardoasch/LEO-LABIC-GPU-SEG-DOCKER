#!/usr/bin/python
# ------------------------- IMPORTS ------------------------- #
import numpy        	 as np
import os, sys, timeit, getopt, cv2, imutils, pickle, scipy
from PIL import Image
PATH = os.path.dirname(os.path.realpath(__file__))
path_lib = './With_Detection/'

sys.path.append(path_lib)
import SSD_detection  	 as ssd

# ------------------------- METHODS ------------------------- #
def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def get_box_image(image, ymin, xmin, ymax, xmax):
	return cv2.cvtColor(image[ymin:ymax, xmin:xmax, :], cv2.COLOR_BGR2RGB)


def get_box_image_label(image, ymin, xmin, ymax, xmax):
	return image[ymin:ymax, xmin:xmax]

def extract(file_name_ori):
	# ------------------------- MAIN ------------------------- #
	# Load detector
	return_dict = ssd.start_interactive_session(ckpt_path=path_lib+'tf_checkpoint/ssd_300_vgg.ckpt',
												allow_growth=True,
												log_device_placement=False)



	image_original = cv2.imread(file_name_ori, cv2.IMREAD_COLOR)


	image_original = imutils.resize(image_original, width=min(600, image_original.shape[1]))

	# Detect bounding boxes
	rclasses, rscores, rbboxes = ssd.process_image(	img             =   image_original,
	                                            	session         =   return_dict['isess']            ,
	                                            	image_4d        =   return_dict['image_4d']         ,
	                                            	predictions     =   return_dict['predictions']      ,
	                                            	localisations   =   return_dict['localisations']    ,
	                                            	bbox_img        =   return_dict['bbox_img']         ,
	                                            	img_input       =   return_dict['img_input']        ,
	                                            	ssd_anchors     =   return_dict['ssd_anchors'])

	# If a person was detected within the frame
	if 15 in rclasses:
		# Get coordinates of the detected bounding box
		'''ymin, xmin, ymax, xmax = ssd.get_bbox_with_people(	classes	=rclasses,
															boxes	=rbboxes,
															height	=image_original.shape[0],
															width	=image_original.shape[1])'''
		boxes = ssd.get_bbox_with_people(	classes	=rclasses,
															boxes	=rbboxes,
															height	=image_original.shape[0],
															width	=image_original.shape[1])
		list_bb = []
		for ymin, xmin, ymax, xmax in boxes:
			print(ymin, xmin, ymax, xmax)
			# Extract the image within the bounding box and apply the classifier
			segmented_image = get_box_image(image_original, ymin, xmin, ymax, xmax)
			
			size = 200,600
			img = Image.fromarray(np.asarray(segmented_image).astype('uint8'), 'RGB')
			#print("ANTES", img)
			img = img.resize(size, Image.NEAREST)
			list_bb.append(img)
		return list_bb
			#print("DEPOIS ", img)
			#img.save(path_images_dest +  '/'+file_name_ori)
		
		
#extract("ex8.jpg")