# -*- coding: utf-8 -*-
'''
@Time    : 2021/7/6 16:05
@Author  : Junfei Sun
@Email   : sjf2002@sohu.com
@File    : drawer.py
'''

import turtle
def draw(predict_result,predictY):
    turtle.penup()
    turtle.goto(-len(predict_result)*3,30)
    for i in predict_result:
        if i==1:
            turtle.pencolor('blue')
            turtle.pensize(9)
            turtle.pendown()
            turtle.forward(6)
        else:
            turtle.penup()
            turtle.forward(6)
    turtle.goto(-len(predict_result)*3,-30)
    for i in predictY:
        if i==1:
            turtle.pencolor('blue')
            turtle.pensize(9)
            turtle.pendown()
            turtle.forward(6)
        else:
            turtle.penup()
            turtle.forward(6)
    turtle.done()