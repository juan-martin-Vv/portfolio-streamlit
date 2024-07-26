import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from io import StringIO
from proces import * 

#IMAGE_FILE = './img/images.jfif'
def face_detect(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    image_orig, image_detec, detection_result = face_detect_buffer(img,flag_2=1)
    return image_orig, image_detec, detection_result
def face_detect_buffer(buffer):
    
    img_buffer = np.fromstring(buffer,np.uint8)
    img1 = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
    if check_image_buffer(buffer=buffer) is -1:
        return None
    img1=buffer_to_img(buffer=buffer)
    #img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    # STEP 2: Create an FaceDetector object.
    base_options = python.BaseOptions(model_asset_path='./blaze_face_short_range.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    # STEP 3: Load the input image.
    image_orig = mp.Image(image_format=mp.ImageFormat.SRGB, data=img1)
    #.Image.create_from_file(IMAGE_FILE)

    # STEP 4: Detect faces in the input image.
    detection_result = detector.detect(image_orig)
    # STEP 5: Process the detection result. In this case, visualize it.
    image_copy = np.copy(image_orig.numpy_view())
    image_detec = visualize(image_copy, detection_result)
    return image_orig, image_detec, detection_result
#image_preces , resul = read_image(IMAGE_FILE)
#print("tipo:")
#print(type(image_preces))
#
#rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
#cv2.imshow("foti",rgb_annotated_image)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()
