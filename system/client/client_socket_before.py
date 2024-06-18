import socket
from _thread import *
import numpy as np
import json
import torch

HOST = '192.9.202.184'
PORT = 12125

client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

data_decode = ""
def recv_data(client_socket) :
    global data_decode
    count = 0
    while True :
        try:
            data = client_socket.recv(100000)

            if not data:
                break
            count += 1
            data_decode += data.decode()

        except ConnectionResetError as e:
            break

        # data_decode += data.decode()
        # print(data.decode())

start_new_thread(recv_data, (client_socket,))
print ('>> Connect Server')


import cv2
import dlib
import part_extraction

shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
image_path = "img.png"

image = cv2.imread(image_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
faces = detector(image)

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


message = str(image.tolist()) + '$' + str(position)
# print(len(message))


def divide_string(text, chunk_size):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
message_chunk = divide_string(message, 100000)
# print(len(message_chunk))
client_socket.send(("!" + str(len(message)) + "!").encode())
for chunk in message_chunk:
    client_socket.send(chunk.encode())


while True:
    message = input('')
    if message == 'quit':
        close_data = message
        break
    elif message == 'first result':
        # print(len(data_decode))
        output2 = json.loads(data_decode)
        output2 = torch.tensor(output2, dtype=torch.float32)
        import usernet

        usernet = usernet.UserNet()
        user_result = usernet.forward(output2)
        print(">>User Ready")
        message2 = str(user_result.tolist())
        # print(len(message2))
        message2_chunk = divide_string(message2, 100000)
        print(len(message2_chunk))
        client_socket.send(("@" + str(len(message2)) + "@").encode())
        for chunk in message2_chunk:
            client_socket.send(chunk.encode())
        data_decode = ""
    elif message == 'result':
        output3 = data_decode
        print(output3)
