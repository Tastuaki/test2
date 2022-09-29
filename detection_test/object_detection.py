import numpy as np
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf
import sys
import time

print(tf.__version__)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
 
PATH_TO_SAVED_MODEL = "C:/Users/ei2113/Documents/ssd_mobilenet_v2_2"
 
print('Loading model...', end='')
start_time = time.time()
 
# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
 
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

image_np = cv2.imread(r"C:/Users/ei2113/Documents/img/reizouko.jpg")
category_index = label_map_util.create_category_index_from_labelmap(r"C:/Users/ei2113/Documents/mscoco_label_map.pbtxt", use_display_name=True)
#category_index = {1: {'id': 1, 'name': 'person'}, 2: {'id': 2, 'name': 'a'}, 3: {'id': 3, 'name': 'surfboard'}, 4: {'id': 4, 'name': 'kite'}}
# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image_np)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'],
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=.40,
    agnostic_mode=False)

resized = cv2.resize(image_np_with_detections, dsize=(1280, 720))
cv2.imwrite("detection_test/detection_test.jpg", resized)
#cv2.waitKey(1) & 0xff
#cv2.destoryAllWindows()