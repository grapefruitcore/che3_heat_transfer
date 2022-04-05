import numpy as np
import matplotlib.pyplot as plt

#Modified from lecture 10 example 2

#BC values 
Tleft=25
Tright=25

#Spacing for y
dy=0.001
(y0,yend)=(0,0.25)
n=int((yend-y0)/dy + 1) #number of points in y
x_val=np.linspace(y0,yend,n)

#spacing for t
dt=100
(t0,tend)=(0,1500)
m=int((tend-t0)/dt + 1) #number of points in t
t_val=np.linspace(t0,tend,m)

#lambda=k delt/delx^2
k=0.00006# this is k/roCp, calculated using literature values (see analytical solution)
lam=k*dt/dy**2

#Initial values of T are uniformly 140 C
T=np.ones((m,n))*140

#BCs
#first column and last column are all 25 C
T[:,0]=Tleft
T[:,-1]=Tright


A=np.diag((1+2*lam)*np.ones(n-2))+np.diag(-lam*np.ones(n-3),1)+\
  np.diag(-lam*np.ones(n-3),-1)
  
#print(A)

#b for the first step
b=np.copy(T[0,1:-1])
b[0]+=lam*Tleft
b[-1]+=lam*Tright

X=np.linalg.solve(A,b)

T[1,1:-1]=X

#b for the rest of the steps
for l in range(1,m-1):
	b=np.copy(T[l,1:-1])
	b[0]+=lam*Tleft
	b[-1]+=lam*Tright
	T[l+1,1:-1]=np.linalg.solve(A,b)

#Plots
plt.figure()
label=[]
for i in range(int((tend/dt)+1)):
    plt.plot(x_val,T[i])
    label.append('t = '+str(i*dt) + ' s')

plt.xlabel("y")
plt.ylabel("T")
plt.title("T as a function of position")

plt.legend(label,bbox_to_anchor=(1.1, 1.05))