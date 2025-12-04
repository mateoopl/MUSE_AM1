import matplotlib.pyplot as plt
from numpy import  array, zeros, concatenate, linalg, any, reshape, linspace, sqrt
import Stability
import Numerical_Schemes



def LinearOscillator(U):                 #Equation definition. U=(x,y)
    F1 = zeros(2)
    F1[0] = U[1] 
    F1[1] = -U[0]
    return F1

def Kepler(U):                                #Kepler's equation
        
    norm = (U[0]**2 + U[1]**2)**0.5
    F1 = zeros(4)
    F1[0] = U[2] 
    F1[1] = U[3]
    F1[2] = -U[0]/norm 
    F1[3] = -U[1]/norm
    return F1
    


def restricted_3bodyProblem_F(U): 

    M1 = 5.972e24  #EARTH MASS
    M2 = 7.348e22  #MOON MASS

    mu = M2/(M1+M2)

    F = zeros(4)

    r1 = sqrt((U[0]+mu)**2 + U[1]**2)
    r2 = sqrt((U[0]-1+mu)**2 + U[1]**2)
 
    F[0] = U[2]
    F[1] = U[3]
    F[2] = 2*U[3] + U[0] - (((1-mu) * (U[0]+mu)) / (r1**3)) - (mu * (U[0]- (1-mu)) / (r2**3))
    F[3] = -2*U[2] + U[1] - (((1-mu) * U[1]) / (r1**3)) - (mu * U[1] / (r2**3))
 

    return F


def NBodyProblem(Nb,Nc,Nv,r0,v0,t,Temporal_scheme): 


    Nt = len(t) 

    U0 = zeros((Nb * Nc * Nv))  
    Us = reshape(U0, (Nb, Nc, Nv))
    r = reshape(Us[:,:,0],(Nb,Nc))  # Positions
    v = reshape(Us[:,:,1], (Nb,Nc))  # Velocities


    for i in range(Nb):    #We build the vector U with the initial conditions
        r[i,:] = r0[i,:]
        v[i,:] = v0[i,:]  

    def Function(X): 
            Xs = reshape(X, (Nb, Nc, Nv))
            xr = reshape(Xs[:,:,0],(Nb,Nc))  # Positions
            xv = reshape(Xs[:,:,1], (Nb,Nc))  # Velocities

            F = zeros((Nb * Nc * Nv))
            Fs = reshape(F, (Nb, Nc, Nv)) 
            drdt = reshape(Fs[:,:,0],(Nb,Nc)) #Velocities
            dvdt = reshape(Fs[:,:,1],(Nb,Nc)) #Accelerations

            for i in range(Nb):
                drdt[i,:] = xv[i,:]
                for j in range(Nb):
                    if i!=j: 
                        distancia_vect =  (xr[j,:]-xr[i,:])
                        dist = linalg.norm(xr[j,:]-xr[i,:])
                        if dist < 1:
                            dvdt[i,:] = dvdt[i,:] + (distancia_vect/(100*dist**3))
                        else :
                            dvdt[i,:] = dvdt[i,:] + (distancia_vect/(dist**3))
            return F
    
    U = Stability.Cauchy_Problem(Function,U0,t,Temporal_scheme)

    U_sol = reshape(U,(Nt,Nb,Nc,Nv))
    rsol = reshape(U_sol[:,:,:,0],(Nt,Nb,Nc))  # Positions
    vsol = reshape(U_sol[:,:,:,1],(Nt,Nb,Nc))  # Positions

    return rsol,vsol


def NBodyProblem_embedded(Nb,Nc,r0,v0,t0,tf,Nv=2): 

    U0 = zeros((Nb * Nc * Nv))  
    Us = reshape(U0, (Nb, Nc, Nv))
    r = reshape(Us[:,:,0],(Nb,Nc))  # Positions
    v = reshape(Us[:,:,1], (Nb,Nc))  # Velocities


    for i in range(Nb):    #We build the vector U with the initial conditions
        r[i,:] = r0[i,:]
        v[i,:] = v0[i,:]
    

    def Function(X): 
        Xs = reshape(X, (Nb, Nc, Nv))
        xr = reshape(Xs[:,:,0],(Nb,Nc))  # Positions
        xv = reshape(Xs[:,:,1], (Nb,Nc))  # Velocities

        F = zeros((Nb * Nc * Nv))
        Fs = reshape(F, (Nb, Nc, Nv)) 
        drdt = reshape(Fs[:,:,0],(Nb,Nc)) #Velocities
        dvdt = reshape(Fs[:,:,1],(Nb,Nc)) #Accelerations

        for i in range(Nb):
            drdt[i,:] = xv[i,:]
            for j in range(Nb):
                if i!=j: 
                    distancia_vect =  (xr[j,:]-xr[i,:])
                    dist = linalg.norm(xr[j,:]-xr[i,:])
                    #if dist < 1:
                    #    dvdt[i,:] = dvdt[i,:] + (distancia_vect/(100*dist**3))
                    #else :
                    dvdt[i,:] = dvdt[i,:] + (distancia_vect/(dist**3))
        return F
    

    U, tiempo, E, dt2, iterations = Numerical_Schemes.Embedded_RK45(U0,t0,tf,Function)

    Nt = len(tiempo)

    U_sol =reshape(U,(Nt,Nb,Nc,Nv))
    rsol = reshape(U_sol[:,:,:,0],(Nt,Nb,Nc))  # Positions
    vsol = reshape(U_sol[:,:,:,1],(Nt,Nb,Nc))  # Positions

    return rsol,vsol

