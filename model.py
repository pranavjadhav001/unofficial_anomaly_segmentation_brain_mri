import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Reshape,Flatten,Dropout,Subtract,Add
from tensorflow.keras.layers import BatchNormalization,Activation,ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU,Lambda
from tensorflow.keras.layers.experimental.preprocessing import Resizing,Rescaling
from tensorflow.keras.layers import UpSampling2D,Conv2D,Conv2DTranspose,MaxPooling2D,AveragePooling2D
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_addons.image.filters import gaussian_filter2d

def create_laplacian_arch(height=256,width=256,channels=3,k=3):
	assert height%32 == 0,"height should be multiple of 32" 
	assert width%32 == 0,"width should be multiple of 32"
	assert k == 3,"laplacian pyramid depth till 3 supported" 
	assert channels==1 or channels==3,'only channels 3 or 1 supported'
	image = Input(shape=(height,width,channels),name='input image')
	down_image1 = Lambda(lambda image:gaussian_filter2d(image,filter_shape=(5,5)))(image)# apply gaussian blur
	down_image1 = Resizing(height//2,width//2,interpolation='bilinear',name='down_image1')(down_image1)#downsample
	down_image2 = Lambda(lambda image:gaussian_filter2d(image,filter_shape=(5,5)))(down_image1)
	down_image2 = Resizing(height//4,width//4,interpolation='bilinear',name='down_image2')(down_image2)
	down_image3 = Lambda(lambda image:gaussian_filter2d(image,filter_shape=(5,5)))(down_image2)
	down_image3 = Resizing(height//8,width//8,interpolation='bilinear',name='down_image3')(down_image3)
	up_image = Resizing(height,width,interpolation="bilinear",name='up_image')(down_image1)
	up_image1 = Resizing(height//2,width//2,interpolation="bilinear",name='up_image1')(down_image2)
	up_image2 = Resizing(height//4,width//4,interpolation="bilinear",name='up_image2')(down_image3)
	down_image1 = Rescaling(scale=1./255)(down_image1)#rescale b/w 0-1
	down_image2 = Rescaling(scale=1./255)(down_image2)
	down_image3 = Rescaling(scale=1./255)(down_image3)
	up_image = Rescaling(scale=1./255)(up_image)
	up_image1 = Rescaling(scale=1./255)(up_image1)
	up_image2 = Rescaling(scale=1./255)(up_image2)
	image_rescaled = Rescaling(scale=1./255)(image)

	subtracted = Subtract()([down_image2,up_image2])#lowest laplacian model i/p
	x = subtracted

	filters = 32
	for i in range(3):# encoder 
	    x = Conv2D(filters=filters,kernel_size=(3,3),strides=2,padding='same',name=f"model2_conv{i}")(x)
	    x = LeakyReLU(alpha=0.2)(x)
	    x = BatchNormalization()(x)
	    filters *= 2

	for i in range(3):# decoder
	    x = Conv2DTranspose(filters=filters,kernel_size=(3,3),strides=2,padding='same',name=f"model2_convtrans{i}")(x)
	    x = LeakyReLU(alpha=0.2)(x)
	    x = BatchNormalization()(x)
	    filters /= 2
	    
	x = Conv2D(filters=channels,kernel_size=(3,3),strides=1,activation='linear',padding='same',name=f"model2_conv_final")(x)
	model2_output = Add()([x, up_image2])# add 
	model2_output_resized = Resizing(height//2,width//2,interpolation='bilinear')(model2_output)

	subtracted = Subtract()([down_image1,up_image1])
	x = subtracted

	filters = 32
	for i in range(4):
	    x = Conv2D(filters=filters,kernel_size=(3,3),strides=2,padding='same',name=f"model1_conv{i}")(x)
	    x = LeakyReLU(alpha=0.2)(x)
	    x = BatchNormalization()(x)
	    filters *= 2

	for i in range(4):
	    x = Conv2DTranspose(filters=filters,kernel_size=(3,3),strides=2,padding='same',name=f"model1_convtrans{i}")(x)
	    x = LeakyReLU(alpha=0.2)(x)
	    x = BatchNormalization()(x)
	    filters /= 2
	    
	x = Conv2D(filters=channels,kernel_size=(3,3),strides=1,activation='linear',padding='same',name=f"model1_conv_final")(x) 
	model1_output = Add()([x, model2_output_resized])
	model1_output_resized = Resizing(height,width,interpolation='bilinear')(model1_output)

	subtracted = Subtract()([image_rescaled,up_image])
	x = subtracted
	filters = 32
	for i in range(5):
	    x = Conv2D(filters=filters,kernel_size=(3,3),strides=2,padding='same',name=f"model_conv{i}")(x)
	    x = LeakyReLU(alpha=0.2)(x)
	    x = BatchNormalization()(x)
	    filters *= 2

	for i in range(5):
	    x = Conv2DTranspose(filters=filters,kernel_size=(3,3),strides=2,padding='same',name=f"model_convtrans{i}")(x)
	    x = LeakyReLU(alpha=0.2)(x)
	    x = BatchNormalization()(x)
	    filters /= 2
	x = Conv2D(filters=channels,kernel_size=(3,3),strides=1,activation='linear',padding='same',name=f"model_conv_final")(x) 
	model_output = Add()([x, model1_output_resized])

	#assign inputs and outputs of model
	model = Model(inputs=image,outputs=[model2_output,down_image2,model1_output,down_image1,model_output,image_rescaled])
	# add loss for each scale of laplacian and sum
	loss = tf.keras.losses.MeanSquaredError()(model2_output, down_image2) +\
	       tf.keras.losses.MeanSquaredError()(model1_output, down_image1) +\
	       tf.keras.losses.MeanSquaredError()(model_output, image_rescaled)
	model.add_loss(loss)
	model.compile(optimizer='adam')
	return model
	