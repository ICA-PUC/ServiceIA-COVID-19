from tqdm import tqdm
from PIL import Image
import glob
from random import random,shuffle,choice
import numpy as np
import cv2
import imutils #rotação imagens
import itertools
from os.path import join

from sklearn.model_selection import train_test_split

from keras.models import Sequential,Model,load_model
from keras.layers import Dense , Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD , RMSprop, Adam
from keras.layers import Conv2D , BatchNormalization
from keras.layers import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.applications import vgg19,mobilenet
from sklearn.preprocessing import OneHotEncoder

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

import ipdb

PATH_xray = 'Dataset/Xray/'
PATH_xray_not_pulmonar = 'Dataset/Xray_not_pulmonar/'
PATH_general = 'Dataset/GeneralImages/'
modelFilesPath="Classificador_treinado_0512_5steps.h5"
USE_PRETRAINED_MODEL = False
SAVED_MODEL_DATA_PATH = "Classificador_treinado_0512.h5"
TENSORBOARD_DIR = "tensorboard_log"

wsize,hsize,channels = 224,224,1

BATCH_SIZE_X_RAY = 0.5
BATCH_SIZE_X_RAY_ROTATED = 0.166
BATCH_SIZE_X_RAY_NOT_PULMONAR = 0.166
BATCH_SIZE_NOT_X_RAY = 0.166

BATCH_X_RAY_VAL = 300
BATCH_X_RAY_ROTATED_VAL = 100
BATCH_X_RAY_NOT_PULMONAR_VAL = 100
BATCH_NOT_X_RAY_VAL = 100
VAL_SIZE = BATCH_X_RAY_VAL+BATCH_X_RAY_ROTATED_VAL+BATCH_X_RAY_NOT_PULMONAR_VAL+BATCH_NOT_X_RAY_VAL

BATCH_X_RAY_TEST = 300
BATCH_X_RAY_ROTATED_TEST = 100
BATCH_X_RAY_NOT_PULMONAR_TEST = 100
BATCH_NOT_X_RAY_TEST = 100
TEST_SIZE = BATCH_X_RAY_TEST+BATCH_X_RAY_ROTATED_TEST+BATCH_X_RAY_NOT_PULMONAR_TEST+BATCH_NOT_X_RAY_TEST

BATCH_SIZE = 128
STEPS_PER_EPOCH = 5
EPOCH_SIZE = BATCH_SIZE*STEPS_PER_EPOCH
EPOCHS = 200

SAVE_IMAGES = False

def print_configurations():
  print(" ---------- configurações do treino ---------------------")
  print("BATCH_SIZE",BATCH_SIZE)
  print("STEPS_PER_EPOCH",STEPS_PER_EPOCH)
  print("TEST_SIZE",TEST_SIZE)
  print("SAVED_MODEL_DATA_PATH",SAVED_MODEL_DATA_PATH)
  print("USE_PRETRAINED_MODEL",USE_PRETRAINED_MODEL)
  print("SAVE_IMAGES",SAVE_IMAGES)
  print(" --------------------------------------------------------")

class train_xray_dataset():

  def __init__(self):
    self.ohe = OneHotEncoder()
    self.ohe.fit([[0],[1],[2],[3],[4]]) #Y_train.reshape(-1,1))

  def get_mobilenet(self):
    """ Criação da Rede - MobileNet """
    print("MobileNet")
    input_shape = (wsize,hsize,channels)
    model = mobilenet.MobileNet(input_shape=input_shape, alpha=1.0, depth_multiplier=1, 
      dropout=1e-3, include_top=False, weights=None, #'imagenet', 
      input_tensor=None, pooling=None)
    x = model.input
    y = model.output
    y = Flatten()(y)
    y = Dense(1 , activation='sigmoid',name="last")(y) #5

    model = Model(input=x,output=y)

    model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.001),
                      metrics=['binary_accuracy'])

    print(model.summary())

    self.model_xray = model

  # ------------------------------- get dataset
  def get_images_list(self,path_group):
      filenames = glob.glob(join(path_group,'*.*'))
      shuffle(filenames)
      return filenames

  def separate_train_val_test(self):
    filenames_xray = self.get_images_list(PATH_xray)
    self.test_xray = filenames_xray[:BATCH_X_RAY_TEST]
    self.val_xray = filenames_xray[BATCH_X_RAY_TEST:BATCH_X_RAY_TEST+BATCH_X_RAY_VAL]
    self.train_xray = filenames_xray[BATCH_X_RAY_TEST+BATCH_X_RAY_VAL:]
    print("imagens X-ray",len(filenames_xray))
    filenames_xray_rotated = self.get_images_list(PATH_xray)
    self.test_xray_rotated = filenames_xray_rotated[:BATCH_X_RAY_ROTATED_TEST]
    self.val_xray_rotated = filenames_xray_rotated[BATCH_X_RAY_ROTATED_TEST:BATCH_X_RAY_ROTATED_TEST+BATCH_X_RAY_ROTATED_VAL]
    self.train_xray_rotated = filenames_xray_rotated[BATCH_X_RAY_ROTATED_TEST+BATCH_X_RAY_ROTATED_VAL:]
    print("imagens X-ray rotated",len(filenames_xray_rotated))    
    filenames_xray_notpulmonar = self.get_images_list(PATH_xray_not_pulmonar)
    self.test_xray_not_pulmonar = filenames_xray_notpulmonar[:BATCH_X_RAY_NOT_PULMONAR_TEST]
    self.val_xray_not_pulmonar = filenames_xray_notpulmonar[BATCH_X_RAY_NOT_PULMONAR_TEST:BATCH_X_RAY_NOT_PULMONAR_TEST+BATCH_X_RAY_NOT_PULMONAR_VAL]
    self.train_xray_not_pulmonar = filenames_xray_notpulmonar[BATCH_X_RAY_NOT_PULMONAR_TEST+BATCH_X_RAY_NOT_PULMONAR_VAL:]
    print("imagens X-ray NOT pulmonar",len(filenames_xray_notpulmonar))
    filenames_not_xray = self.get_images_list(PATH_general)
    self.test_not_xray = filenames_not_xray[:BATCH_NOT_X_RAY_TEST]
    self.val_not_xray = filenames_not_xray[BATCH_NOT_X_RAY_TEST:BATCH_NOT_X_RAY_TEST+BATCH_NOT_X_RAY_VAL]
    self.train_not_xray = filenames_not_xray[BATCH_NOT_X_RAY_TEST+BATCH_NOT_X_RAY_VAL:]
    print("imagens NOT X-ray",len(filenames_not_xray))

  def generate_Train_dataset(self,batch_size):
    
    #filenames_xray,filenames_xray_notpulmonar,filenames_not_xray = self.train_xray,self.train_xray_not_pulmonar,self.train_not_xray

    count_batches = 0

    while True:

        if count_batches % (STEPS_PER_EPOCH) == 0:
            #copys to avoid using smae images
            epoch_list_xray = self.train_xray.copy()
            shuffle(epoch_list_xray)
            epoch_list_xray_rotated = self.train_xray_rotated.copy()
            shuffle(epoch_list_xray_rotated)
            epoch_list_xray_notpulmonar = self.train_xray_not_pulmonar.copy()
            shuffle(epoch_list_xray_notpulmonar)
            epoch_list_not_xray = self.train_not_xray.copy()
            shuffle(epoch_list_not_xray)
          
        x_total,y_total = self.get_batch(epoch_list_xray,epoch_list_xray_rotated,epoch_list_xray_notpulmonar,epoch_list_not_xray,batch_size)

        count_batches+=1

        yield x_total,y_total
  
  def get_batch(self,epoch_list_xray,epoch_list_xray_rotated,epoch_list_xray_notpulmonar,epoch_list_not_xray,batch_size):

        #print("-> imagens X-ray")
        images_xray,epoch_list_xray = self.get_dataset_group(epoch_list_xray,int(BATCH_SIZE_X_RAY*batch_size))
        images_xray = self.data_augmentation(images_xray,max_angle_rotation=5,max_zoom=False,square_angles=False)
        label_images_xray = np.ones(len(images_xray), dtype=np.int8)
        #print("imagens X-ray",images_xray.shape,label_images_xray.shape,np.unique(label_images_xray))

        #print("-> imagens X-ray rotated")
        images_xray_rotated,epoch_list_xray_rotated = self.get_dataset_group(epoch_list_xray_rotated,int(BATCH_SIZE_X_RAY_ROTATED*batch_size))
        images_xray_rotated,_ = self.data_augmentation(images_xray_rotated,max_angle_rotation=5,max_zoom=0.1,square_angles=True)
        label_images_xray_rotated =  np.zeros(len(images_xray_rotated), dtype=np.int8)
        #print("imagens X-ray rotated",images_xray_rotated.shape,label_images_xray_rotated.shape,np.unique(label_images_xray_rotated))

        # x-xay not pulmonar
        #print("-> imagens X-ray NOT pulmonar")
        images_xray_notpulmonar,epoch_list_xray_notpulmonar = self.get_dataset_group(epoch_list_xray_notpulmonar,int(BATCH_SIZE_X_RAY_NOT_PULMONAR*batch_size))
        images_xray_notpulmonar,_ = self.data_augmentation(images_xray_notpulmonar,max_angle_rotation=5,max_zoom=False,square_angles=True)
        label_images_xray_notpulmonar =  np.zeros(len(images_xray_notpulmonar), dtype=np.int8)
        #print("imagens X-ray NOT pulmonar",images_xray_notpulmonar.shape,label_images_xray_notpulmonar.shape,np.unique(label_images_xray_notpulmonar))

        # non X-ray
        #print("-> imagens NOT X-ray")
        images_general,epoch_list_not_xray = self.get_dataset_group(epoch_list_not_xray,int(BATCH_SIZE_NOT_X_RAY*batch_size))
        y_general = np.zeros(len(images_general), dtype=np.int8)
        #print("imagens NOT X-ray",images_general.shape,y_general.shape,np.unique(y_general))

        # join x-Ray + non x-rR
        x_total = np.concatenate((images_xray,images_xray_rotated), axis=0)
        x_total = np.concatenate((x_total,images_xray_notpulmonar), axis=0)
        x_total = np.concatenate((x_total,images_general), axis=0)
        y_total = np.concatenate((label_images_xray,label_images_xray_rotated), axis=0)
        y_total = np.concatenate((y_total,label_images_xray_notpulmonar), axis=0)
        y_total = np.concatenate((y_total,y_general), axis=0)
        #print(x_total.shape,y_total.shape)
        #print("Balanceamento classe:",np.unique(y_total,return_counts=True))
    
        #y_total_ohe = np.array(self.one_hot_encoder(y_total))

        # Preprocessamento das imagens (resize and gray)
        x_total = self.preprocess_dataset(x_total,False)

        # corret format (add axis)
        x_total = self.training_format(x_total)

        return x_total,y_total

  def get_dataset_group(self, path_group,batch_size_group):
    images_group = []
    count=0
    for filename in path_group:
        im=self.make_image_square(cv2.imread(filename))
        images_group.append(im)
        count+=1
        if count>int(batch_size_group):
          break
    path_group = path_group[batch_size_group:]
    
    return np.array(images_group),path_group

  def make_image_square(self,image):
    if not image.shape[0]==image.shape[1]:
      lateral_size = min(image.shape[0],image.shape[1])
      image_xray_square = cv2.resize(image,(lateral_size,lateral_size))
      return image_xray_square
    else:
      return image

  def one_hot_encoder(self,label):
    label_ohe = self.ohe.transform(label.reshape(-1,1)).toarray()
    return label_ohe
  
  # ------------------------------------  
  def save_images(self, images,prefix):
    print("saving images:",len(images))
    count=1
    for img in images:
      cv2.imwrite('dataset_analisys/'+prefix+str(count)+".png", img) 
      count+=1
    print("Done saving images")
    
  # ------------------ data augmentation
  def data_augmentation(self,images,max_angle_rotation=False,max_zoom=False,square_angles=False):
    #print(" - Data Augmentation:")
    if max_angle_rotation:
      #print(" rotation,random - max angle:",max_angle_rotation)
      images = [self.Image_random_rotation(img,max_angle_rotation) for img in images]
    if max_zoom:
      #print(" zoom,random - max zoom:",max_zoom)
      images = [self.zoom_image(img,max_zoom) for img in images]
    if square_angles:
      #print(" square_angles,random:",max_zoom)
      images,labels = self.multiImages_X_Ray_square_rotation(images)
      return images,labels
    
    return np.array(images)

  def zoom_image(self,image,max_zoom):
    zoom = random()*max_zoom
    maxx = image.shape[0]
    maxy = image.shape[1]
    new_xmin = int(maxx*zoom/2)
    new_ymin = int(maxy*zoom/2)
    new_xmax = int(maxx-new_xmin)
    new_ymax = int(maxy-new_ymin)
    image_zoomed = image[new_xmin:new_xmax, new_ymin:new_ymax]
    return image_zoomed

  def Image_random_rotation(self,image,max_angle):
    angle = (random()-0.5)*max_angle*2
    rotated = imutils.rotate_bound(image, angle)
    return rotated
  
  def multiImages_X_Ray_square_rotation(self,images):
    multiImages = []
    multiImages_y = []
    for img in images:
      x,y = self.Image_X_Ray_square_rotation(img)
      multiImages.append(x)
      multiImages_y.append(y)
    return np.array(multiImages),np.array(multiImages_y)

  def Image_X_Ray_square_rotation(self,image):
    possible_angles = [90,180,270]
    angle = choice(possible_angles)
    image = imutils.rotate_bound(image, angle)
    for i in range(len(possible_angles)):
        if angle == possible_angles[i]:
            label = 1 #i+1

    return image,label

  # -----------------------------------------------
  def preprocess_dataset(self,images,save_image=False):
    '''
    convert imagens:
    1) gray
    2) resize para tamanho predefinido
    '''
    # gray and resize
    resized_and_gray = np.array([cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),(hsize,wsize)) for img in images])
    if save_image:
      self.save_images(resized_and_gray,"resized_and_gray_")

    return resized_and_gray
  
  def training_format(self,images):
    '''
    put dataset int the corret format for trainning: adds extra axis in the end
    '''
    return images[:,:,:,np.newaxis]/255
    #print("Dataset size and shape:",images.shape)
      
  def split_train_test(self):
      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x_total, self.y_total, test_size=0.1, random_state=42)

  def train_model(self):
    # Treinamento do autoencoder
    early_stopping = EarlyStopping(monitor='val_loss', patience=15)
    checkpoint = ModelCheckpoint(filepath=modelFilesPath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
    reduceLRonPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto')
    tensorboard_call = TensorBoard(log_dir=TENSORBOARD_DIR)

    self.model_xray.fit_generator(self.generate_Train_dataset(BATCH_SIZE),
        validation_data=(self.x_val, self.y_val), 
        steps_per_epoch=STEPS_PER_EPOCH,
        verbose=1,
        epochs=EPOCHS,
        shuffle=True,
        callbacks=[early_stopping,checkpoint,reduceLRonPlateau,tensorboard_call])

  def test_model(self):

    trained_model = load_model(modelFilesPath)

    out = trained_model.predict(self.x_test)

 
    certos_xray = sum(out[:BATCH_X_RAY_TEST]>0.5)
    certos_xray_rotated = sum(out[BATCH_X_RAY_TEST:BATCH_X_RAY_TEST+BATCH_X_RAY_ROTATED_TEST]<0.5)
    certos_xray_not_pulmonar = sum(out[BATCH_X_RAY_TEST+BATCH_X_RAY_ROTATED_TEST:BATCH_X_RAY_TEST+BATCH_X_RAY_ROTATED_TEST+BATCH_X_RAY_NOT_PULMONAR_TEST]<0.5)
    certos_not_xray = sum(out[
        BATCH_X_RAY_TEST+BATCH_X_RAY_ROTATED_TEST+BATCH_X_RAY_NOT_PULMONAR_TEST:
        BATCH_X_RAY_TEST+BATCH_X_RAY_ROTATED_TEST+BATCH_X_RAY_NOT_PULMONAR_TEST+BATCH_NOT_X_RAY_TEST]<0.5)

    print("certos_xray:",certos_xray,certos_xray/BATCH_X_RAY_TEST)
    print("certos_xray_rotated:",certos_xray_rotated,certos_xray_rotated/BATCH_X_RAY_ROTATED_TEST)
    print("certos_xray_not_pulmonar:",certos_xray_not_pulmonar,certos_xray_not_pulmonar/BATCH_X_RAY_NOT_PULMONAR_TEST)
    print("certos_not_xray:",certos_not_xray,certos_not_xray/BATCH_NOT_X_RAY_TEST)

    totais_certos = certos_xray + certos_xray_rotated + certos_xray_not_pulmonar + certos_not_xray
    print("-> totais certos:",totais_certos,totais_certos/len(out))
    print(self.y_test)
    print(np.array(out)[:,0])

        
if __name__ == "__main__":

  # instanciar classe
  x_ray_train = train_xray_dataset()

  # get model
  if not USE_PRETRAINED_MODEL:
    x_ray_train.get_mobilenet()
  else:
    x_ray_train.model_xray=load_model(SAVED_MODEL_DATA_PATH)

  print_configurations()

  # separar test de train
  x_ray_train.separate_train_val_test()

  # get val dataset
  print(" ---------- get val data ----------")
  x_ray_train.x_val,x_ray_train.y_val = x_ray_train.get_batch(x_ray_train.val_xray,x_ray_train.val_xray_rotated,x_ray_train.val_xray_not_pulmonar,x_ray_train.val_not_xray,VAL_SIZE)
  print("val data:",x_ray_train.x_val.shape,x_ray_train.y_val.shape,np.unique(x_ray_train.y_val,return_counts=True))

  # get test dataset
  print(" ---------- get test data ----------")
  x_ray_train.x_test,x_ray_train.y_test = x_ray_train.get_batch(x_ray_train.test_xray,x_ray_train.test_xray_rotated,x_ray_train.test_xray_not_pulmonar,x_ray_train.test_not_xray,TEST_SIZE)
  print("test data:",x_ray_train.x_test.shape,x_ray_train.y_test.shape,np.unique(x_ray_train.y_test,return_counts=True))

  # train
  x_ray_train.train_model()

  # analyse test data
  x_ray_train.test_model()


