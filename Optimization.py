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
from bayes_opt import BayesianOptimization
import drawer
import random



def load_data():

    with open("./Data/Total.pkl","br") as fh:
        data=pickle.load(fh)

    trainX=data[0]
    predictX=data[1]
    trainY=data[2]
    predictY=data[3]


    trainX=trainX.reshape((trainX.shape[0],11,5,1))
    predictX = predictX.reshape((predictX.shape[0], 11,5,1))
    '''
     print('trainX:',trainX.shape)
    print('trainY:', trainY.shape)
    '''


    trainY=to_categorical(trainY)

    return trainX,trainY,predictX,predictY


def define_model(Nf,Sf,Nf2,Sf2,Nhl,Nhn,Afi,lr):
    Nf=int(round(Nf))
    Sf=int(round(Sf))
    Nf2=int(round(Nf2))
    Sf2=int(round(Sf2))
    Nhl=int(round(Nhl))
    Nhn=int(round(Nhn))
    Afi=int(round(Afi))
    Af=['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu']
    Af=Af[Afi]
    model=Sequential()
    model.add(Conv2D(Nf, (Sf,Sf),activation='relu',kernel_initializer='he_uniform',input_shape=(11,5,1)))
    if Sf2<=6-Sf:
        model.add(Conv2D(Nf2, (Sf2, Sf2), activation='relu', kernel_initializer='he_uniform', input_shape=(12-Sf, 6-Sf, 1)))
    model.add(Flatten())
    for i in range(Nhl):
        model.add(Dense(Nhn, activation=Af, kernel_initializer='he_uniform'))
    model.add(Dense(2, activation='softmax'))

    opt=SGD(lr=lr, momentum=0.9)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    #print(model.summary())

    return model


def Optimize():

    param={
        'Nf':(2,10),
        'Sf':(2,3),
        'Nf2':(2,10),
        'Sf2':(2,3),
        'Nhl':(1,4),
        'Nhn':(5,200),
        'Afi':(0,5),
        'lr':(0,1)
    }
    op_model=BayesianOptimization(f=evaluate_model,pbounds=param)

    op_model.maximize(init_points=25, n_iter=15)
    params= op_model.max['params']
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu']
    params['Afi'] = activationL[int(round(params['Afi']))]
    print(params)


def evaluate_model(Nf,Sf,Nf2,Sf2,Nhl,Nhn,Afi,lr,nfolds=5):
    scores,histories=list(),list()
    dataX,dataY,predictX,predictY=load_data()
    kfold=KFold(nfolds,shuffle=True,random_state=1)
    for train_ix, test_ix in kfold.split(dataX):
        model=define_model(Nf,Sf,Nf2,Sf2,Nhl,Nhn,Afi,lr)
        trainX,trainY,testX,testY=dataX[train_ix],dataY[train_ix],dataX[test_ix],dataY[test_ix]
        history=model.fit(trainX,trainY,epochs=10,batch_size=60,validation_data=(testX,testY),verbose=0)
        #print(history.history.keys())
        _,acc=model.evaluate(testX,testY,verbose=0)
        #print('>%.3f'%(acc*100))
        scores.append(acc)
        histories.append(history)
    '''
    print("scores",scores)
    print("histories.len",len(histories))
    '''
    accuracy=0
    for i in scores:
        accuracy=accuracy+i
    accuracy=accuracy/len(scores)

    return accuracy

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


def main():
    #trainX,trainY,predictX,predictY=load_data()
    Optimize()
    #summarize_plotter(histories)
    #summarize_performance(scores)
    #predict_result=predict(model,predictX,predictY)
    #drawer.draw(predict_result,predictY)





#main()
load_data()
main()