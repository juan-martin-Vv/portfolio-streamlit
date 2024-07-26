import cv2
import math
import mediapipe as mp
#########################
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white


# Create the options that will be used for ImageSegmenter
base_options = python.BaseOptions(model_asset_path='./deeplab_v3.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)

# Height and width that will be used by the model
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
#IMAGE_FILENAMES = ['./img/perfil.jpg']
# Performs resizing and showing the image
def resize_and_show(image):
  if image is None:
    print("param passed is Null")
    return -1
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2.imshow("name",img)


def imag_segmeter(image_orig, form_buff=0):    
    if image_orig is None:
       return None
    #image_orig=cv2.cvtColor(image_orig,cv2.COLOR_BGR2RGB)
    if form_buff==1:
        img_buffer = np.fromstring(image_orig,np.uint8)
        img1 = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        image_orig = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    output_image=[]
    # Create the image segmenter
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        if segmenter is None:
           return None
      # Loop through demo image(s)
      #for image_file_name in IMAGE_FILENAMES:

        # Create the MediaPipe image file that will be segmented
       # image = mp.Image.create_from_file(image_file_name)
        image=mp.Image(image_format=mp.ImageFormat.SRGB,data=image_orig)
        # Retrieve the masks for the segmented image
        segmentation_result = segmenter.segment(image)
        category_mask = segmentation_result.category_mask

        # Generate solid color images for showing the output segmentation mask.
        image_data = image.numpy_view()
        fg_image = np.zeros(image_data.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR


        # Apply effects
        blurred_image = cv2.GaussianBlur(image_data, (55,55), 0)
        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
        output_image.append( np.where(condition, image_data, blurred_image) )

        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) < 0.4
        output_image.append( np.where(condition, bg_image, image_data))

        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) < 0.2
        output_image.append( np.where(condition, image_data, fg_image))

        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
        output_image.append(np.where(condition, fg_image, bg_image))
        #print(f'Segmentation mask of {name}:')
        return output_image
        #resize_and_show(output_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
# Preview the image(s)

#images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
#for name, image in images.items():
#  print(name)
#  result=imag_segmeter(image)
#  if result is not None:
#    resize_and_show(image=image)
#    cv2.imshow("original",result[3])
#    cv2.imshow("fondo",result[1])
#    cv2.imshow("plano",result[2])
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
