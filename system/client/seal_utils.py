import sys
sys.path.append('/home/hwlee/senier/system')
from SEAL_Python.examples.seal import *
import numpy as np
import json

class Seal():
    def __init__(self):
        # BGV 암호화 스킴을 사용할 암호화 매개변수 객체를 생성합니다.
        parms = EncryptionParameters(scheme_type.bgv)

        poly_modulus_degree = 8192
        parms.set_poly_modulus_degree(poly_modulus_degree)
        parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
        parms.set_plain_modulus(8192)

        context = SEALContext(parms)

        # 키 생성
        keygen = KeyGenerator(context)

        secret_key = keygen.secret_key()
        public_key = keygen.create_public_key()
        relin_keys = keygen.create_relin_keys()
        galois_keys = keygen.create_galois_keys()

        self.encryptor = Encryptor(context, public_key)
        self.evaluator = Evaluator(context)
        self.decryptor = Decryptor(context, secret_key)
        
    def encryption(self, image, position, height, width):
        print(height)
        print(width)
        img = []
        for i in range(height):
            for j in range(width):
                if position[i][j] == 1:
                    # image[i][j][0] = self.encryptor.encrypt(Plaintext(image[i][j][0]))
                    # image[i][j][1] = self.encryptor.encrypt(Plaintext(image[i][j][1]))
                    # image[i][j][2] = self.encryptor.encrypt(Plaintext(image[i][j][2]))
                    img.append(self.encryptor.encrypt(Plaintext(image[i][j][0])))
                    img.append(self.encryptor.encrypt(Plaintext(image[i][j][1])))
                    img.append(self.encryptor.encrypt(Plaintext(image[i][j][2])))
                    # print(image[i][j][0])
                else:
                    img.append(image[i][j][0])
                    img.append(image[i][j][1])
                    img.append(image[i][j][2])
        
        return img, self.decryptor

    def decryption(self, image_encryption):
        return 1