from numpy import mean
from numpy import std
import numpy as np
import pickle
from matplotlib import pyplot
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import drawer
import pylab

def load_data():

    with open("./Data/Total.pkl","br") as fh:
        data=pickle.load(fh)

    trainX=data[0]
    predictX=data[1]
    trainY=data[2]
    predictY=data[3]


    trainX=trainX.reshape((trainX.shape[0],11,5,1))
    predictX = predictX.reshape((predictX.shape[0], 11,5,1))
    print('trainX:',trainX.shape)
    print('trainY:', trainY.shape)

    trainY=to_categorical(trainY)

    return trainX,trainY,predictX,predictY


def define_model():
    model=Sequential()
    model.add(Conv2D(6, (3,3),activation='relu',kernel_initializer='he_uniform',input_shape=(11,5,1)))
    model.add(Conv2D(3, (2,2),activation='relu', kernel_initializer='he_uniform', input_shape=(9,3,8)))

    model.add(Flatten())
    model.add(Dense(128,activation='softsign',kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='softsign', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='softsign', kernel_initializer='he_uniform'))
    model.add(Dense(2, activation='softmax'))

    opt=SGD(lr=0.0449, momentum=0.9)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    print(model.summary())
    return model

def evaluate_model(dataX,dataY,nfolds=5):
    scores,histories=list(),list()

    kfold=KFold(nfolds,shuffle=True,random_state=1)
    for train_ix, test_ix in kfold.split(dataX):
        model=define_model()
        trainX,trainY,testX,testY=dataX[train_ix],dataY[train_ix],dataX[test_ix],dataY[test_ix]
        history=model.fit(trainX,trainY,epochs=30,batch_size=60,validation_data=(testX,testY),verbose=0)
        print(history.history.keys())
        _,acc=model.evaluate(testX,testY,verbose=0)
        print('>%.3f'%(acc*100))
        scores.append(acc)
        histories.append(history)
    print("scores",scores)
    print("histories.len",len(histories))
    return scores,histories,model

def summarize_plotter(histories):
    for i in range(len(histories)):
        #loss
        pyplot.subplot(2,1,1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'],color='blue',label='train')
        pyplot.plot(histories[i].history['val_loss'], color='red', label='test')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train','test'],loc='upper right')

        #accuracy
        pyplot.subplot(2,1,2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='red', label='test')
        pyplot.ylabel('accuracy')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'test'], loc='upper right')

    pyplot.show()

def summarize_performance(scores):
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    pyplot.boxplot(scores)
    pyplot.show()

def predict(model,predictX,predictY):
    predict_result=model.predict(predictX)
    predict_result=[1 if p[1]>=0.5 else 0 for p in predict_result]
    total=0
    correct=0
    wrong=0
    print('predictresult:',predict_result)
    print('predictY:',predictY)
    for i in range(len(predict_result)):
        total+=1
        if predictY[i]==predict_result[i]:
            correct+=1
        else:
            wrong+=1
    print('accuracy:',correct/total)
    return predict_result

def visualization(model,image):
    layer_names = [layer.name for layer in model.layers]
    layer_outputs = [layer.output for layer in model.layers]

    feature_map_model = tf.keras.models.Model(model.input, layer_outputs)
    input = image.reshape((1,) + image.shape)
    feature_maps = feature_map_model.predict(input)
    for layer_name, feature_map in zip(layer_names, feature_maps):
        if len(feature_map.shape) == 4:  # Number of feature images/dimensions in a feature map of a layer
            k = feature_map.shape[-1]
            height = feature_map.shape[1]
            width = feature_map.shape[2]
            print('height:',height,'width:',width)
            image_belt = np.zeros((height, width* k))
            for i in range(k):
                feature_image = feature_map[0, :, :, i]

                feature_image -= feature_image.mean()
                feature_image /= feature_image.std()
                feature_image *= 64
                feature_image += 128

                feature_image = np.clip(feature_image, 0, 255).astype('uint8')

                image_belt[:, i * width: (i + 1) * width] = feature_image
            scale = 20. / k
            pyplot.figure(figsize=(scale * k, scale))
            pyplot.title(layer_name)
            pyplot.grid(False)
            pyplot.imshow(image_belt, aspect='auto')
            pylab.show()


def main():
    trainX,trainY,predictX,predictY=load_data()
    scores,histories,model=evaluate_model(trainX,trainY)
    summarize_plotter(histories)
    summarize_performance(scores)
    predict_result=predict(model,predictX,predictY)
    drawer.draw(predict_result,predictY)
    '''
    visualizeimg=np.array([[0, 0, 0, 0, 0],
 [0, 1, 0, 0, 0],
 [1, 1, 0, 0, 0],
 [1, 1, 1, 0, 1],
 [1, 1, 1, 1, 1],
 [0, 0, 0, 0, 0],
 [1, 1, 1, 1, 1],
 [1, 1, 1, 0, 1],
 [1, 1, 0, 0, 0],
 [0, 1, 0, 0, 0],
 [0, 0, 0, 0, 0]])
    visualization(model,visualizeimg)
    '''





load_data()
main()
