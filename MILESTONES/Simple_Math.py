import matplotlib.pyplot as plt
from numpy import  array, zeros, concatenate, linalg, any, linspace

def derivative(f,x,dx):
    h = 1e-4
    return( (f(x+dx)- f(x-dx)) /(2*h ))

def Newton_escalar(f,x):
    h = 1e-4
    while (abs(f(x)/derivative(f,x,h)) > 1e-8):
        x = x - f(x)/derivative(f,x,h)
    return(x)

def Jacobian(f,x):
    N = len(x)
    J = zeros((N,N))
    for j in range(N):
        dx = zeros(N)
        dx[j] = 1e-4
        J[:,j] = derivative(f,x,dx)
    return J


def Newton(f,x0):

    J = Jacobian(f,x0)
    inv_J = linalg.inv(J)
    while (linalg.norm(inv_J@f(x0)) > 1e-8):
        x0 = x0 - inv_J@f(x0)
        J = Jacobian(f,x0)
        inv_J = linalg.inv(J)
    return(x0)