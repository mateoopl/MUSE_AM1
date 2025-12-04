import matplotlib.pyplot as plt
from numpy import  array, zeros, concatenate, linalg, any, linspace, shape, log
import Numerical_Schemes
import Simple_Math


def Cauchy_Problem(F,U0,t,Temporal_scheme):  #F: Funcion Rn,  #U0: vector Rn, #T: Tiempo
                                            
    Nt = len(t) - 1     #Número de intervalos temporales
    Nv = len(U0)         #Número de variables
    U = zeros((Nt+1,Nv))
    U[0,:] = U0

    if Temporal_scheme == Numerical_Schemes.LeapFrog:         #For a LeapFrog, we need also U[n-1] for each step. We add a new step on a different loop.
        U[1,:] = Numerical_Schemes.Euler(U[0,:],t[0],t[1],F)
        for n in range(1,Nt):
            U[n+1,:] = Numerical_Schemes.LeapFrog(U[n,:],U[n-1,:],t[n],t[n+1],F)    #Respetamos la API
    else:
        for n in range(Nt):
            U[n+1,:] = Temporal_scheme(U[n,:],t[n],t[n+1],F)    #Respetamos la API
    return U




def Cauchy_Error(F,U0,t,Temporal_scheme,q=1):  #F: Funcion Rn,  #U0: vector Rn, #t: Tiempo

    Nt = len(t) - 1
    Nv = len(U0)
            #Nt, Nv = shape(U1)
    E = zeros((Nt+1,Nv))

    t1 = t[:]
    
    t2 = linspace(t[0], t[Nt], 2*(Nt)+1 )
   

    U1 = Cauchy_Problem(F,U0,t1,Temporal_scheme)  #Solucion del problema de Cauchy en la malla 1
    U2 = Cauchy_Problem(F,U0,t2,Temporal_scheme)  #Solucion del problema de Cauchy en la malla 2


    for n in range(Nt):
        E[n,:] = (U2[2*n,:] - U1[n,:])/(1-1/2**q)

    return U1, E



def Convergence_rate(Temporal_scheme, F, U0,t, Nm=10, q=1):


    N_mesh = zeros(Nm,dtype=int)
    E = zeros(Nm)
    logN = zeros(Nm)
    logE = zeros(Nm)
   
    for i in range(Nm):
        N_mesh[i] = (20 + 20*i)
    # N_mesh = array([100, 300, 500, 700, 900])

    
    N = len(t) - 1
    for n in range(Nm):
        t_n = linspace(t[0], t[N], N_mesh[n])
        U1, E = Cauchy_Error(F,U0,t_n,Temporal_scheme, q)
        logN[n] = log(len(t_n))
        logE[n] = log(abs(E).max())

    deltay = logE[int(Nm/2) + 1] - logE[int(Nm/2)]
    deltax = logN[int(Nm/2) + 1] - logN[int(Nm/2)]

    m = deltay/deltax

    return logN, logE, m