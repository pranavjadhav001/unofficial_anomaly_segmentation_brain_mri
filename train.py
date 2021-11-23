from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_laplacian_arch 

model = create_laplacian_arch(256,256)
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(r'D://python3/mvtec_anomaly_detection/bottle/train',color_mode='rgb',\
                                 class_mode='input',batch_size=4,shuffle=True,target_size=(256,256))

model.fit(train_generator,epochs=100,steps_per_epoch=len(train_generator))