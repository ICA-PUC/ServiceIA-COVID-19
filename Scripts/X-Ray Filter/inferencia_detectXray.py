from keras.models import load_model
import numpy as np
#import pandas as pd
import os
import cv2
import glob
from random import shuffle
from termcolor import colored

from classificadorimagensxray_generator_2classes import *


MODEL_DATA_PATH = "Classificador_treinado_0408_2.h5"
wsize,hsize,channels = 224,224,1
normalizacao = 255

x_ray_train = train_xray_dataset()


def execute_one_inference(image_path,debug=False):
    '''
    execute one image inference. returns probability value (close to 1 -> X-Ray image)
    '''
    if debug:
        print("----------------- Load Model:",MODEL_DATA_PATH)
    newModel = load_model(MODEL_DATA_PATH)

    # read image
    image = np.array([preprocess_image(cv2.imread(image_path))])/normalizacao

    # inference image
    res = inference_images(newModel,image,debug)
    
    if debug:
        diagnostic = "Não é X-Ray"
        if res>0.5:
            diagnostic = "é X-Ray"
            print(image_path,"results:",res," -> ",colored(diagnostic,"yellow"))
        else:
            print(image_path,"results:",res," -> ",colored(diagnostic,"blue"))


    return res


def inference_images(model,images,debug=False): 
    
    if debug:
        print(images.shape)
    out = model.predict(images)

    return out

def preprocess_image(image,zoom=False, rotate=False, square_rotate=False):
    '''
    prepare image for inference:
    resize and gray color
    '''
    if zoom:
        image = x_ray_train.zoom_image(image,0.1)
    if rotate:
        image = x_ray_train.Image_random_rotation(image,5)
    if square_rotate:
        image,_ = x_ray_train.Image_X_Ray_square_rotation(image)

    resized_and_gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),(hsize,wsize))
    
    return resized_and_gray[:,:,np.newaxis]

def execute_multiple_images(images_path,zoom=False, rotate=False, square_rotate=False):

    # read images
    num_imagens=3000
    print("images_path:",images_path)
    filenames = glob.glob(images_path)
    shuffle(filenames)
    images = np.array([preprocess_image(cv2.imread(img),zoom,rotate,square_rotate) for img in filenames[:num_imagens]])/normalizacao
    print(" - TOTAL FILES:", len(images)) #,"1a imagem:",images[0][0].shape)
    if len(images) == 0:
        exit()

    # inference images
    res = inference_images(newModel,images)
    count_xRay=0
    count_notXrayray=0
    count=0
    for r in res:
        diagnostic = "Não é X-Ray"
        if r>0.5:
            diagnostic = "é X-Ray"
            count_xRay+=1
            print(filenames[count],"results:",r," -> ",colored(diagnostic,"yellow"))
        else:
            count_notXrayray+=1
            print(filenames[count],"results:",r," -> ",colored(diagnostic,"blue"))


        print("results:",r," -> ",diagnostic)
        print("X-Ray:",count_xRay, "Not X-Ray:",count_notXrayray)
        count+=1
    
    return count_xRay,count_notXrayray

# test only
if __name__ == "__main__":
    
    #execute_one_inference("Dataset/Xray/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg",True)

    print("----------------- Load Model:",MODEL_DATA_PATH)
    newModel = load_model(MODEL_DATA_PATH)

    r1,nr1 = execute_multiple_images("Dataset/Xray/*",zoom=False, rotate=True, square_rotate=False) #*S1684*")
    r2,nr2 = execute_multiple_images("Dataset/Xray/*",zoom=True, rotate=True, square_rotate=True) #*S1684*")
    r3,nr3 = execute_multiple_images("Dataset/GeneralImages/*",zoom=False, rotate=False, square_rotate=True) #2011_00384*")
    r4,nr4 = execute_multiple_images("Dataset/Xray_not_pulmonar/*",zoom=False, rotate=True, square_rotate=True) # *11*")

    print("x-ray pulmonar",r1,nr1)
    print("x-ray pulmonar rotated",r2,nr2)
    print("outros",r3,nr3)
    print("x-ray outros",r4,nr4)




