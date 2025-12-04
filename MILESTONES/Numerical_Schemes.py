import matplotlib.pyplot as plt
from numpy import  array, zeros, concatenate, linalg, any, reshape, linspace
import Simple_Math


def Euler(U,t1,t2,F):      

    dt = t2 - t1
    return U + dt*F(U)


def RangeKutta4(U,t1,t2,F):

    dt = t2 - t1
    k1 = F(U)
    k2 = F(U + (dt/2)*k1)
    k3 = F(U + (dt/2)*k2)
    k4 = F(U + (dt)*k3)
    return U + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


def Inverse_Euler(U,t1,t2,F):
    dt = t2-t1
    def G(x):
        return x - U - dt*F(x)
    return Simple_Math.Newton(G,U)



def Crank_Nicolson(U,t1,t2,F):
    dt = t2-t1
    a = U + (dt/2)*F(U)
    def G(x):
        return x - a - (dt/2)*F(x)
    return Simple_Math.Newton(G,U)


def LeapFrog(U,U1,t1,t2,F):
    dt = t2-t1
    return U1 + 2*dt*F(U)

def RangeKutta45(U,t1,t2,F):

        dt = t2 - t1

        k1 = dt*F(U)
        k2 = dt*F(U + (2/9)*k1)
        k3 = dt*F(U + (1/12)*k1 + (1/4)*k2)
        k4 = dt*F(U + (69/128)*k1 + (-243/128)*k2 + (135/64)*k3)
        k5 = dt*F(U + (-17/12)*k1 + (27/4)*k2 + (-27/5)*k3 + (16/15)*k4)
        k6 = dt*F(U + (65/432)*k1 + (-5/16)*k2 + (13/16)*k3 + (4/27)*k4 + (5/144)*k5)
        
        U1 = U + (47/450)*k1 + (0)*k2 + (12/25)*k3 + (32/225)*k4 + (1/30)*k5 + (6/25)*k6
        Error = linalg.norm(k1*((47/450)-(1/9)) + k2*((0)-(0)) + k3*((12/25)-(9/20)) + k4*((32/225)-(16/45)) + k5*((1/30)-(1/12)) + k6*((6/25)-(0)))

        return U1, Error


def Embedded_RK45(U0,t1,tf,F,dt_0=0.1,tol = 1e-6):
    Error_0 = 0.0

    dt_values = [dt_0]      #We define lists to store the values: variable size!
    t_values = [t1]        
    U_values = [U0]         
    E_values = [Error_0]    

    

    t2 = t1 + dt_0
    i = 0
    dt = dt_0
    

    while t1 < tf:
        
        U, Error  = RangeKutta45(U_values[i],t1,t2,F)       #Calculate U and E in i+1
       

        if Error < 1e-16:   
            Error = 1e-16
         
        if Error <= tol:    
            U_values.append(U)
            t_values.append(t2)
            E_values.append(Error)
            dt_values.append(dt)
            i += 1
            #print(f"Step {i+1}: t = {t2}, dt = {dt}, Error = {Error}")

            t1 = t_values[i]
            t2 = t1 + dt_values[i]

        else:                                    
            dt_values[i] = dt_values[i]*(tol/Error)**(1/5)
            dt = dt_values[i]
            t1 = t_values[i]
            t2 = t1 + dt
            #print("!!!!"f"Step {i+1}: t = {t2}, dt = {dt}, Error = {Error}")
            
    return array(U_values), array(t_values), array(E_values), array(dt_values),i   



