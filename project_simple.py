import numpy as np
import matplotlib.pyplot as plt
#Group: Yu An Chen, Michael Orrett, ABM Hussein, Evelyn Gondosiswanto
#Modified from che322 lecture 10 example 2

#BC values room temperature
Tleft=25
Tright=25

#Spacing for y, super small because my laptop has lots of processing power
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
#first column and last column are all 25 C because boundaries
T[:,0]=Tleft
T[:,-1]=Tright

#A
A=np.diag((1+2*lam)*np.ones(n-2))+np.diag(-lam*np.ones(n-3),1)+\
  np.diag(-lam*np.ones(n-3),-1)

#b loop for each t step
for l in range(0,m-1):
	b=np.copy(T[l,1:-1])
	b[0]+=lam*Tleft
	b[-1]+=lam*Tright
	T[l+1,1:-1]=np.linalg.solve(A,b)

#Plot
plt.figure()
label=[]
#loop to plot each time
for i in range(int((tend/dt)+1)):
    plt.plot(x_val,T[i])
    label.append('t = '+str(i*dt) + ' s')

plt.xlabel("y")
plt.ylabel("T")
plt.title("T as a function of position")

#moving the legend to the right to not block the graph
plt.legend(label,bbox_to_anchor=(1.1, 1.05))

#"Funny how I'm always on the head of the arrow."