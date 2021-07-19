# -*- coding: utf-8 -*-
'''
@Time    : 2021/6/29 10:48
@Author  : Junfei Sun
@Email   : sjf2002@sohu.com
@File    : DataProcessing.py
'''
import numpy as np
import os
import math
import csv
'''
file_name = input('Please input the file name')
output_name=input('Please input the output file name')
os.chdir("Data/")
'''

unitlength = 5

def make_data(file_name, output_name):
    flag = False
    f = open(file_name, 'r')
    data = f.readlines()
    f.close()
    counter = 0
    for line in data:
        counter += 1
        line = line.strip('\n')
        line = line.strip()
        line = line.split(' ')
        while '' in line:
            line.remove('')
        print(line)

        if counter==1:

            Sequence_length=eval(line[1])
            matrix = np.array(range(1, Sequence_length * 12 + 1)).reshape(12, Sequence_length)
            for i in range(1, 13):
                for j in range(1, Sequence_length + 1):
                    matrix[i - 1][j - 1] = 0
            print(Sequence_length)

        elif counter >= 2 and line[0] != 'END' and not flag:
            if abs(eval(line[1]) - eval(line[0])) <= 5:
                matrix[eval(line[0]) - eval(line[1]) + 5][eval(line[0]) - 1] = 1
                matrix[eval(line[1]) - eval(line[0]) + 5][eval(line[0]) - 1] = 1
        elif line[0] == "END":
            if matrix.shape[1] % unitlength == 0:
                ArrMatrix = np.hsplit(matrix, Sequence_length / unitlength)
            else:
                ArrMatrix = np.hsplit(matrix[0:12, 0:matrix.shape[1] - matrix.shape[1] % unitlength],
                                      (matrix.shape[1] - matrix.shape[1] % unitlength) / unitlength)
                print(len(ArrMatrix))
            flag = True
        elif flag:
            tempc=0
            newline=[]
            for i in range(0,len(line)):
                if line[i+1]=='?' and tempc==0:
                    newline.append(line[i])
                    tempc+=1
                elif line[i+1]=='?' and tempc==1:
                    newline.append(line[i])
                    tempc+=1
                elif tempc>=2:
                    break
            line=newline
            print(line)
            if math.ceil(eval(line[1]) / unitlength)<= len(ArrMatrix):
                for index in range(math.floor(eval(line[0]) / unitlength) ,math.ceil(eval(line[1]) / unitlength)):
                    ArrMatrix[index][11][4]=1
                    print(ArrMatrix[index])
            else:
                for index in range(math.floor(eval(line[0]) / unitlength), len(ArrMatrix)):
                    print(index)
                    ArrMatrix[index][11][4]=1
                    print(ArrMatrix[index])
    file=open(output_name,'w')
    ft=open("Dataset.csv",'a')
    for i in ArrMatrix:
       csvwriter=csv.writer(ft)
       csvwriter.writerow(i.flatten())
       np.savetxt(file,i)

    ft.close()
    file.close()


def predict_data(prefile):
    os.chdir("Predict/")
    flag = False
    f = open(prefile, 'r')
    data = f.readlines()
    f.close()
    counter = 0
    for line in data:
        counter += 1
        line=line.replace('\t',' ')
        line = line.strip('\n')
        line = line.strip()
        line = line.split(' ')
        while '' in line:
            line.remove('')

        if counter == 1:
            Sequence_length = eval(line[1])
            matrix = np.array(range(1, Sequence_length * 12 + 1)).reshape(12, Sequence_length)
            for i in range(1, 13):
                for j in range(1, Sequence_length + 1):
                    matrix[i - 1][j - 1] = 0
            print(Sequence_length)
        elif counter >= 2 and line[0] != 'END' and not flag:
            if abs(eval(line[2]) - eval(line[0])) <= 5 and eval(line[len(line)-1])>0.85:
                print(line)
                matrix[eval(line[0]) - eval(line[2]) + 5][eval(line[0]) - 1] = 1
                matrix[eval(line[2]) - eval(line[0]) + 5][eval(line[0]) - 1] = 1
        elif line[0] == "END":
            if matrix.shape[1] % unitlength == 0:
                ArrMatrix = np.hsplit(matrix, Sequence_length / unitlength)
            else:
                ArrMatrix = np.hsplit(matrix[0:12, 0:matrix.shape[1] - matrix.shape[1] % unitlength],
                                      (matrix.shape[1] - matrix.shape[1] % unitlength) / unitlength)
                print(len(ArrMatrix))
            flag = True
        elif flag:
            tempc = 0
            newline = []
            for i in range(0, len(line)):
                if line[i + 1] == '?' and tempc == 0:
                    newline.append(line[i])
                    tempc += 1
                elif line[i + 1] == '?' and tempc == 1:
                    newline.append(line[i])
                    tempc += 1
                elif tempc >= 2:
                    break
            line = newline
            print(line)
            if math.ceil(eval(line[1]) / unitlength) <= len(ArrMatrix):
                for index in range(math.floor(eval(line[0]) / unitlength), math.ceil(eval(line[1]) / unitlength)):
                    ArrMatrix[index][11][4] = 1
                    print(ArrMatrix[index])
            else:
                for index in range(math.floor(eval(line[0]) / unitlength), len(ArrMatrix)):
                    print(index)
                    ArrMatrix[index][11][4] = 1
                    print(ArrMatrix[index])
    ft = open("Predictset.csv", 'a')
    for i in ArrMatrix:
        csvwriter = csv.writer(ft)
        csvwriter.writerow(i.flatten())
    ft.close()

#make_data(file_name, output_name)


prefile=input('predicted input name')
predict_data(prefile)

