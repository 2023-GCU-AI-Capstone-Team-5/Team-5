import socket
from _thread import *
import numpy as np
import torch
import json


client_sockets = []


HOST = '192.9.202.184'
PORT = 12125


print('>> Server Start')
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()


def threaded(client_socket, addr):
    print('>> Connected by :', addr[0], ':', addr[1])

    count1 = -1
    count2 = -1
    data_decode = ""
    while True:

        try:

            data = client_socket.recv(100000)

            if not data:
                print('>> Disconnected by ' + addr[0], ':', addr[1])
                break
            
            data_temp = data.decode()
            print(data_temp)
            if data_temp[0] == "!":
                data_split = data_temp[1:].split('!')
                count1 = int(data_split[0])
                print(count1)
                data_decode += data_split[1]
            elif data_temp[0] == "@":
                data_split = data_temp[1:].split('@')
                count2 = int(data_split[0])
                print(count2)
                data_decode += data_split[1]
            else:
                data_decode += data_temp
            
            if len(data_decode) == count1:
                print(len(data_decode))
                data_split = data_decode.split('$')
                image, position = json.loads(data_split[0]), json.loads(data_split[1])
                image = np.array(image)
                image = image[np.newaxis, ...]
                        
                        
                import first_layer
                net_first = first_layer.First_layer(position=position)
                result_first = net_first.forward(image)
                result_first = torch.tensor(result_first, dtype=torch.float32)
                print('>> First Ready')
                
                import utils
                message = str(result_first.tolist())
                print(len(message))
                message_chunk = utils.divide_string(message, 100000)
                print(len(message_chunk))
                for chunk in message_chunk:
                        client_socket.send(chunk.encode())
                data_decode = ""
                count1 = -1
                
            elif len(data_decode) == count2:
                output3 = json.loads(data_decode)
                output3 = torch.tensor(output3, dtype=torch.float32)
                import remain_layer
                result_pred = remain_layer.Remain_layer(output3)
                print(result_pred)
                
                message = str(result_pred)
                client_socket.send(message.encode())
                count2 = -1
                

        except ConnectionResetError as e:
            print('>> Disconnected by ' + addr[0], ':', addr[1])
            break
    print(len(data_decode))
    

    if client_socket in client_sockets :
        client_sockets.remove(client_socket)
        print('remove client list : ',len(client_sockets))

    client_socket.close()


try:
    while True:
        print('>> Wait')

        client_socket, addr = server_socket.accept()
        client_sockets.append(client_socket)
        start_new_thread(threaded, (client_socket, addr))
        print("Number of participant : ", len(client_sockets))
        
except Exception as e :
    print ('Error? : ',e)

finally:
    server_socket.close()
    
    
