import cv2
import dlib
import part_extraction

shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
image_path = "img25.jpg"

image = cv2.imread(image_path)
print(image.shape)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
faces = detector(image)
print(faces)


for face in faces:
    shape = predictor(image, face)
    shape = part_extraction.shape_to_np(shape)
    left_eye = shape[36:42]
    right_eye = shape[42:48]
    nose = shape[27:36]
    mouth = shape[48:68]
    highlighted_image, highlighted_points = part_extraction.highlight_key_parts(image, [left_eye, right_eye, nose, mouth])

height = image.shape[0]
width = image.shape[1]
position = part_extraction.Point_expansion(highlighted_points, height, width)


import seal_utils
SEAL = seal_utils.Seal()
image, decryptor, evaluator = SEAL.encryption(image.tolist(), position, height, width)


import numpy as np
image = np.array(image)
image = image.reshape(1, 592, 474, 3)
print(image.shape)


import torch
import first_layer
net_first = first_layer.First_layer(position=position, evaluator=evaluator)
result_encryption, output_position = net_first.forward(image)

print("dda")
result_decryption = SEAL.decryption(result_encryption, output_position)
# exit(0)
print(result_decryption)
result_decryption = torch.tensor(result_decryption, dtype=torch.float32)


import usernet
usernet = usernet.UserNet()
user_result = usernet.forward(result_decryption)
print(user_result.shape)


import remain_layer
result_pred = remain_layer.Remain_layer(user_result)
print(result_pred)