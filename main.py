from numpy import mean
from numpy import std
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

def load_data():

    with open("./Data/Total.pkl","br") as fh:
        data=pickle.load(fh)

    trainX=data[0]
    testX=data[1]
    trainY=data[2]
    testY=data[3]

    trainX=trainX.reshape((trainX.shape[0],11,5,1))
    testX = testX.reshape((testX.shape[0], 11,5,1))
    print('trainX:',trainX.shape)
    print('trainY:', trainY.shape)

    trainY=to_categorical(trainY)
    testY=to_categorical(testY)

    for i in range(5):
        print("trainY",trainY[i])

    return trainX,trainY,testX,testY


def define_model():
    model=Sequential()
    model.add(Conv2D(8,(2,2),activation='relu',kernel_initializer='he_uniform',input_shape=(11,5,1)))
    model.add(Conv2D(8,(2,2),activation='relu', kernel_initializer='he_uniform', input_shape=(10,4,1)))
    #model.add(MaxPool2D(2,2))
    model.add(Flatten())
    model.add(Dense(59,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(2,activation='softmax'))

    opt=SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    print(model.summary())
    return model

def evaluate_model(dataX,dataY,nfolds=5):
    scores,histories=list(),list()

    kfold=KFold(nfolds,shuffle=True,random_state=1)
    for train_ix, text_ix in kfold.split(dataX):
        model=define_model()
        trainX,trainY,testX,testY=dataX[train_ix],dataY[train_ix],dataX[text_ix],dataY[text_ix]
        history=model.fit(trainX,trainY,epochs=10,batch_size=60,validation_data=(testX,testY),verbose=0)
        print(history.history.keys())
        _,acc=model.evaluate(testX,testY,verbose=0)
        print('>%.3f'%(acc*100))
        scores.append(acc)
        histories.append(history)
    print("scores",scores)
    print("histories.len",len(histories))
    return scores,histories

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

def main():
    trainX,trainY,testX,testY=load_data()
    scores,histories=evaluate_model(trainX,trainY)
    summarize_plotter(histories)
    summarize_performance(scores)

#main()
load_data()
main()