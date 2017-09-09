# import library
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import pandas as pd

# data augment core process
def random_brightness(image, angle):
    """ Variation of brightness """

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    coef = np.random.uniform(0.15, 1.0)
    image[:, :, 2] = image[:, :, 2] * coef
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image, angle


def flip_process(image, angle):
    """ flip image """

    image_flipped = np.fliplr(image)
    angle_flipped = -angle;
    return image_flipped, angle_flipped

def through_process(image, angle):
    """ No processing"""
    return image, angle


def rotate_image(image, angle):
    """rorate image"""

    r_angle = random.randint(-10,10)
    size = tuple(np.array([image.shape[1], image.shape[0]]))    
    M = cv2.getRotationMatrix2D((size[0]/2,size[1]/2),r_angle,1)
    image = cv2.warpAffine(image, M, size, borderMode=cv2.BORDER_REPLICATE)
    return image, angle


def shadow(image, angle):
    """ Random shadow generator"""

    size = tuple(np.array([image.shape[1], image.shape[0]]))  
    [btm_x, top_x] = np.random.choice(size[0], 2, replace=False)
    slope = size[1] / (btm_x - top_x)
    i_cept = -slope * top_x
    coef = brightness_factor = np.random.uniform(0.5, 0.7)
    for y_idx in range(size[1]):
        x_idx = int((y_idx - i_cept) / slope)
        image[y_idx, :x_idx, :] = (image[y_idx, :x_idx, :] * coef).astype(np.int32)        
        return image, angle


def create_list_for_removing_data():
    """ analyze data for training and remove histgram peaking """

    # read trainning data from cvs file
    columns = ['center','left','right','steering','throttle','brake','speed']
    data_array = pd.read_csv(DATA_PATH, names=columns, header=None)

    # create histgram and calculate average data number of bins
    bin_num = 51	# Quantization number of steering			
    angles = data_array['steering']
    ave_num = len(data_array['steering'])/bin_num # average data num
    histogram, bins = np.histogram(data_array['steering'], bin_num) # create histgram

    # create data list for removal
    non_removal_ratio = []	# non removal probability 
    coef = 0.9			# coef for adjustment
    target_num = ave_num * coef # target number of data 

    # calc non removal probability per bin
    for i in range(bin_num):
        if histogram[i] < target_num:
            non_removal_ratio.append(1)
        else:
            non_removal_ratio.append(target_num/histogram[i])

    # create list to remove data
    remove_list = []
    for i in range(len(data_array['steering'])):
        for j in range(bin_num):
            if((j == 0 and data_array['steering'][i] <= bins[j+1]) or (data_array['steering'][i] > bins[j] and data_array['steering'][i] <= bins[j+1])):
                if np.random.rand() > non_removal_ratio[j]:
                    remove_list.append(i)
    data_array.drop(data_array.index[remove_list], inplace=True)

    return remove_list

#import lib
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D, Dropout


def nvidia():
    """ This model is a modified model proposed by Nvidia.  """
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(100, 200, 3)))   
#    model.add(Cropping2D(cropping=((60,20), (0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
#    model.add(Dropout(0.7))
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


def generator(samples, batch_size=32):
    """Generator of images and steering angles for training"""
    
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # process batch sample
                images_data, angles_data = process_line(batch_sample)

                images.extend(images_data)
                angles.extend(angles_data)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)


def generage_data(image, angle):
    """data augment main process"""
    images1 = []
    angles1 = []    

    crp_img = image[60:image.shape[0]-20,:]
    
    img, agl = through_process(crp_img, angle)
    rsz_img = cv2.resize(img,(200,100), interpolation=cv2.INTER_AREA)
    images1.append(rsz_img)
    angles1.append(agl)
   
    idx = random.randint(0, 4)
    if(idx == 0):
        img, agl = random_brightness(crp_img, agl)
        rsz_img = cv2.resize(img,(200,100), interpolation=cv2.INTER_AREA)
        images1.append(rsz_img)
        angles1.append(agl)
    elif(idx == 1):
        img, agl = shadow(crp_img, agl)
        rsz_img = cv2.resize(img,(200,100), interpolation=cv2.INTER_AREA)
        images1.append(rsz_img)
        angles1.append(agl)
    elif(idx == 2):
        img, agl = rotate_image(crp_img, agl)
        rsz_img = cv2.resize(img,(200,100), interpolation=cv2.INTER_AREA)
        images1.append(rsz_img)
        angles1.append(agl)
       
    return images1, angles1



def process_line(line):
    """read line from cvs file"""
    images = []
    angles = []

    image_dir = './data/IMG/'
    correction = [0, 0.2, -0.2]

    for i in range(3):
        # read the image
        # org image          
        image_path = image_dir + line[i].split('/')[-1]
        
        image = mpimg.imread(image_path)
        angle = float(line[3]) + correction[i]        
        imgs, agls = generage_data(image, angle)
        images.extend(imgs)
        angles.extend(agls)

        # flip the image
        image_flipped, angle_flipped = flip_process(image, angle)
        imgs, agls = generage_data(image_flipped, angle_flipped)
        images.extend(imgs)
        angles.extend(agls)
               
    return images, angles

DATA_PATH = './data/driving_log.csv'		#Data log file path


def read_lines_from_file(file_path):
    """Read from CSV file and return lines"""
    lines = []
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file)
#        next(reader, None) # skip the header
        for line in reader:
            lines.append(line)
        return lines

def remove_process(remove_list, samples):
    """ Process to remove peak from histogram """

    idx = len(remove_list)

    # execute remove process
    cnt = 0 
    for i in range(idx):
        del samples[remove_list[i]-cnt]
        cnt +=1

    return samples


# main process
print("Start Creating Removal List!")
remove_list = create_list_for_removing_data()
print("Create Removal List Done!")

# read image and create data set for training
samples = read_lines_from_file(DATA_PATH)		
samples = remove_process(remove_list, samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print(len(samples))
print(len(train_samples))
print(len(validation_samples))
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)
print("Create Training data Done!")


# create model and execute training
epoch_num = 20
samples_epoch = 2 * 3 * 2 * len(train_samples)                # flip * cam_num * coef * data_num
samples_epoch_validation = 2 * 3 * 2 *len(validation_samples) # flip * cam_num * coef * data_num

model = nvidia()
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch= samples_epoch,
                                     validation_data=validation_generator,
                                     nb_val_samples= samples_epoch_validation,
                                     nb_epoch=epoch_num, verbose=1)


# Visualize loss
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("log_fig.png")
plt.show()

# save model
model.save('model.h5')

