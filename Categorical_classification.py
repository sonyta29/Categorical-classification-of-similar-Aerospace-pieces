#we have imported the necessary libraries and firmworks that can provide us high level functions 
import matplotlib
matplotlib.use("Agg")
from keras.preprocessing import image
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import keras
import pickle
import cv2
import os
from PIL import Image


# In[13]:


#Enter the path of your image data folder
image_data_folder_path = "E:/sonyta/neural networks/dataset/dataset2"

# initialize the data and labels as an empty list 
#we will reshape the image data and append it in the list-data
#we will encode the image labels and append it in the list-labels
data = []
labels = []


# In[14]:


#grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(image_data_folder_path)))


# In[15]:


#total number images
total_number_of_images = len(imagePaths)
print("\n")
print("Total number of images----->",total_number_of_images)


# In[16]:


#randomly shuffle all the image file name 
random.shuffle(imagePaths)


# In[17]:


#loop over the shuffled input images
for imagePath in imagePaths:

	#Read the image into a numpy array using opencv
	#all the read images are of different shapes
	image = cv2.imread(imagePath)

	#resize the image to be 32x32 pixels (ignoring aspect ratio)
	#After reshape size of all the images will become 32x32x3
	#Total number of pixels in every image = 32x32x3=3072
	image = cv2.resize(image, (32, 32))

	#flatten converts every 3D image (32x32x3) into 1D numpy array of shape (3072,)
	#(3072,) is the shape of the flatten image
	#(3072,) shape means 3072 columns and 1 row
	image_flatten = image.flatten()

	#Append each image data 1D array to the data list
	data.append(image_flatten)

	# extract the class label from the image path and update the
	label = imagePath.split(os.path.sep)[-2]

	#Append each image label to the labels list
	labels.append(label)
# scale the raw pixel intensities to the range [0, 1]
#convert the data and label list to numpy array
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


# In[18]:


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
# train_test_split is a scikit-learn's function which helps us to split train and test images kept in the same folders
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=20)

print("Number of training images--->",len(trainX),",","Number of training labels--->",len(trainY))
print("Number of testing images--->",len(testX),",","Number of testing labels--->",len(testY))

# convert the labels from integers to vectors 
# perform One hot encoding of all the labels using scikit-learn's function LabelBinarizer
# LabelBinarizer fit_transform finds all the labels 
lb = preprocessing.LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print("\n")
print ("Classes found to train",)
train_classes = lb.classes_
print(train_classes)
binary_rep_each_class = lb.transform(train_classes)
print("Binary representation of each class")
print(binary_rep_each_class)
print("\n")


# In[21]:


# define the 3072-1024-512-3 architecture using Keras
model = Sequential()
# we construct our neural network architecture — a 3072-1024-512-3 feedforward neural network.

# Our input layer has 3072 nodes, one for each of the 32 x 32 x 3 = 3072 raw pixel intensities in our flattened input images
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(train_classes), activation="softmax"))

print ("Printing the summary of model")
model.summary()


# In[22]:


# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 75

# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])


# In[23]:


# train the neural network
H = model.fit(trainX, trainY, validation_data=(testX, testY),epochs=EPOCHS, batch_size=32)


# In[25]:


print("[INFO] serializing network and label binarizer...")
model.save("simple_multiclass_classifcation_model.model")
f = open("simple_multiclass_classifcation_lb.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()


# In[26]:


from keras.models import load_model
model = load_model("simple_multiclass_classifcation_model.model")


# In[27]:


model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])


# In[ ]:



from keras.preprocessing import image
from keras.models import load_model
import argparse
import pickle
import cv2
import os
import matplotlib.pyplot as plt
#test_image= image.load_img("C:/Users/Ghazi Abbes/Desktop/panda.jpg", target_size = (30, 30)) 
test_image_path = "E:/sonyta/neural networks/dataset/test"
model_path = "/simple_multiclass_classifcation_model.model"
label_binarizer_path = "/simple_multiclass_classifcation_lb.pickle"

model = load_model("simple_multiclass_classifcation_model.model")
pickle.dump(lb, open("save.txt", "wb"))
with open("save.txt", "wb") as f:
    pickle.dump(lb, f)
with open("save.txt", "rb") as f:
    lb = pickle.load(f)



images = []
output=[]
for filename in os.listdir(test_image_path):
        img = cv2.imread(os.path.join(test_image_path,filename))
        output = img.copy()
        img = cv2.resize(img, (32,32))
        img = img.astype("float") / 255.0
        img = img.flatten()
        img = img.reshape((1, img.shape[0]))
        preds = model.predict(img)
        #print ("preds.argmax(axis=1)",preds.argmax(axis=1))
        i = preds.argmax(axis=1)[0]
        #print (i)
        label = lb.classes_[i]
        text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
        cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        	(0, 0, 255), 2)
        cv2.imshow("Image", output)
        cv2.waitKey(0)


