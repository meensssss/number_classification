import numpy as np
import os
from os.path import dirname, join, realpath
import joblib
from io import BytesIO
from PIL import Image
#from tensorflow.keras.datasets import mnist
import cv2


'''
# Load the MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Check data dimension
print("X_train data shape is", X_train.shape)
print("y_train data shape is", y_train.shape)
print("X_test  data shape is", X_test.shape )
print("y_test  data shape is", y_test.shape )

# Check data type
print("X_train data type is", X_train.dtype)
print("y_train data type is", y_train.dtype)
print("X_test  data type is", X_test.dtype )
print("y_test  data type is", y_test.dtype )

# Visualise sample data
img_index = 999
print(y_train[img_index])
#plt.imshow(y_train[13])
#plt.show()

# Reshaping the array to 3-d for Keras
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Specify input shape for the mode's 1st layer
input_shape = (28, 28, 1)

# Convert data type to float so we can get decimal points after division
# Also scope down X_train and X_test to 2000:400 samples
X_train = X_train[:2000, :, :, :].astype('float32')
X_test = X_test[:400, :, :, :].astype('float32')

# Also scope down y_train and y_test to 2000:400 samples
y_train = y_train[:2000]
y_test = y_test[:400]

# Normalising all data by dividing by max brightness value.
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print('X_test shape:' , X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:' , y_test.shape)



def x_predict(no_index):
    print("number: ", no_index)
    pred = clf.predict(X_test[int(no_index)].reshape(1, 28, 28, 1))
    print("Prediction probability array is:")
    count = 0

    for i in pred.squeeze():
        print(count, ":", i)
        count += 1

    print("From which the max choice is:", pred.argmax())
    #plt.imshow(X_test[int(no_index)])
    #plt.show()
    return pred.argmax()

#load model from pickle file to test    
with open(
    join(dirname(os.path.realpath(os.path.dirname(__file__))), "model_pipeline.pkl"), "rb"
) as f:
    clf = joblib.load(f)



def load_model():
    path = os.path.join(os.path.dirname(__file__), "model_pipeline.pkl")
    model = joblib.load(path)
    return model

clf = load_model()


def image_predict(images: Image.Image):
     arr= []
#def image_predict(path):
#    images = read_imagepath(path)
     pred = clf.predict(images)
     print("Prediction probability array is:")
     count = 0
     _tem= 0
     for i in pred.squeeze():
        print("X :", count, ":", i)
        if i > 0.80:
            _tem = i
        count += 1

     print("From which the max choice is:", pred.argmax())
     print("From which the probability is:", _tem)
     if _tem < 0.80:
         ans="Can_not_detection"
     else:
         ans=pred.argmax()

     arr.append(ans)
     arr.append(_tem)
     print("Length of List:",len(arr))
     print("List[0]:", arr[0])
     print("List[1]:", arr[1])
     return arr
'''

def read_imagefile(file) -> Image.Image:
    #image  = Image.open(BytesIO(file))
    np_arr = np.fromstring(file, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    image = np.array(image)
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    #image = 255-image
    image /= 255
    print('Data type : ', type(image))
    return image

'''
def image_path_predict(path):
     images = read_imagepath(path)
     pred = clf.predict(images)
     print("Prediction probability array is:")
     count = 0

     for i in pred.squeeze():
        print(count, ":", i)
        count += 1

     print("From which the max choice is:", pred.argmax())
     return pred.argmax()
'''

def read_imagepath(path) -> Image.Image:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = np.array(image)
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    image = 255-image
    image /= 255

    return image


'''
def store_picture_to_db(tags, conf, file): 
    import pyodbc
    import base64
    #connection to store the MSSQL DB.
    conn_str = ("Driver={SQL Server};"
            "Server=KRDANB2210207\SQLEXPRESS;"
            "Database=master;"
            "Trusted_Connection=yes;")
    conn = pyodbc.connect(conn_str)
    
    cursor = conn.cursor()
    if (cursor):
        #save binary file
        #with open(images, 'rb') as photo_file:
        #    photo_bytes = photo_file.read()
        #insert picture into db
        print('Data type : ', type(file))
        
        cursor.execute("INSERT INTO diseases_data_storages (byte_images, disease_type, confidence ) VALUES (?, ?, ?)", pyodbc.Binary(file), str(tags), str(conf))
        cursor.commit()
        cursor.close()
    conn.close()

    return True

'''

#image_path = "C:/Users/rujirang.w/Downloads/MNIST-JPG-master/MNIST-JPG-master/MNIST Dataset JPG format/MNIST Dataset JPG format/MNIST - JPG - testing/0/441.jpg"
#store_picture_to_db(image_path, 'Test')

'''
read_imagepath(image_path)
image_path_predict(image_path)
x_predict(3)
image_predict(image_path)
'''