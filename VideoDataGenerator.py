import numpy as np
import random
import os
from keras.preprocessing.image import load_img, img_to_array
from sklearn import preprocessing
from sklearn.utils import shuffle

class VideoDataGenerator:
    def __init__(self,
                rescale=1./255,
                input_shape=None,
                validation_split=0.2,
                shuffle=True):
        self.frames, self.width, self.height, self.channles = input_shape
        self.rescale = rescale
        self.validation_split = validation_split
        self.shuffle = shuffle


    def samples(self, subset):
        dir = self.dir
        videos = len([video for directory in os.listdir(dir) for video in os.listdir(os.path.join(dir, directory))])
        if subset == "validation":
            return int(self.validation_split * videos)
        elif subset == "training":
            return int((1-self.validation_split) * videos)

    def __generate_batch(self, batch_size, dir, videos, labels, subset, target_size):
        lower_bound = None
        upper_bound = None
        if subset == "validation":
            lower_bound = int(len(videos)*(1-self.validation_split)) + 1
            upper_bound = int(len(videos))
        elif subset == "training":
            lower_bound = 0
            upper_bound = int(len(videos)*(1-self.validation_split))
        data_split = videos[lower_bound : upper_bound]
        labels_split = labels[lower_bound : upper_bound]
        index = 0
        while True:
            batch = []
            batch_labels = []
            for i in range(batch_size):
                path = os.path.join(dir, data_split[index])
                frames = [load_img(os.path.join(path, frame), target_size=target_size) for frame in os.listdir(os.path.join(dir, path))]
                frames = [img_to_array(frame) for frame in frames]
                batch.append(frames)
                batch_labels.append(labels_split[index])
                index += 1
                if index >= len(data_split):
                    if(self.shuffle):
                        data_split, labels_split = shuffle(data_split, labels_split)
                    index = 0
            batch = np.array(batch, dtype="float32")
            batch_labels = np.array(batch_labels, dtype="float32").reshape(-1,)
            if self.rescale != None:
                batch *= self.rescale
            
            rand=random.randint(0, len(data_split))
            yield batch, batch_labels
                
    def flow_from_directory(self,
                            dir=None,
                            subset="training",
                            target_size=(150, 150),
                            batch_size=32):
        self.dir = dir
        videos = np.array([os.path.join(directory, video) for directory in os.listdir(dir) for video in os.listdir(os.path.join(dir, directory))])
        if self.shuffle:
            np.random.shuffle(videos)
        lb = preprocessing.LabelBinarizer()
        labels = np.array(lb.fit_transform([x.split("/")[0] for x in videos]))
        return self.__generate_batch(batch_size, dir, videos, labels, subset, target_size)


