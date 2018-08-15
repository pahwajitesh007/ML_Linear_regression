# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:09:29 2018

@author: JITESH
"""
from numpy import *

#time step to learn our model and reduce our error

def compute_error_for_given_points(b,m,points):
    totalError=0
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        totalError += (y-(m*x+b))**2
    return totalError/ float(len(points))
    

def step_gradient_descent(b_current,m_current,points,learning_rate):
    #gradient descent
    b_gradient=0
    m_gradient=0
    N=float(len(points))

# =============================================================================
#     print('b_current67===',b_current)
#     print('m_current-99===',m_current)
# =============================================================================
    for i in range(0, len(points)):
        x=points[i,0]
        y=points[i,1]
# =============================================================================
#         print('m_current== ',m_current)
#         print('b_current== ',b_current)
# =============================================================================
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
# =============================================================================
#         print('b_gra == ',b_gradient,'b_current== ',b_current,'learning_rate * b_gradient== ',learning_rate * b_gradient)
#         print('m_gra == ',m_gradient,'m_current== ',m_current,'learning_rate * m_gradient == ',learning_rate * m_gradient)
# =============================================================================
    new_b=b_current-(learning_rate * b_gradient)
    new_m=m_current-(learning_rate * m_gradient)
    return[new_b,new_m]
    


def gradient_descent_runner(points,starting_b,starting_m,learning_rate,num_iteration):
    b = starting_b
    m = starting_m
    for i in range(num_iteration):
# =============================================================================
#         print(b,m)
# =============================================================================
        b,m=step_gradient_descent(b,m,array(points),learning_rate)
# =============================================================================
#         print('b == ',b)
#         print('m == ',m)
# =============================================================================
    return[b,m]
    
def run():
    points=genfromtxt('data.csv',delimiter=',')
    # hyper permaters for tuning our model. how fast our model learning 
    learning_rate=0.0001
    #y=mx+c(slope formula)
    initial_b=0
    initial_m=0
    num_iteration=10
   # print('points== ',points[:100])
    [b,m]=gradient_descent_runner(points[:100],initial_b,initial_m,learning_rate,num_iteration)
    print('After {0} iterations b ={1} , m ={2}, error= {3}'.format(num_iteration,b,m,compute_error_for_given_points(b,m,points)))
    
    
if __name__== '__main__':
    run()