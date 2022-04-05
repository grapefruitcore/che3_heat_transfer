import numpy as np
import matplotlib.pyplot as plt

#Modified from lecture 10 example 2

#BC values 
Tleft=25
Tright=25

#spacing for x
#try changing this afterwards
dy=0.001
(y0,yend)=(0,0.25)
n=int((yend-y0)/dy + 1) #number of points in x
x_val=np.linspace(y0,yend,n)


#spacing for t
dt=100
(t0,tend)=(0,1500)
m=int((tend-t0)/dt + 1) #number of points in t
t_val=np.linspace(t0,tend,m)


#lambda=k delt/delx^2
k=0.00006# this is k/roCp
lam=k*dt/dy**2

#row 1 of T holds temperature values at t0=0
#row 2 holds values at t=t0+dt
#row 3 holds values at t=t0+2*dt
#we need m rows, n columns
#could have been other way around
T=np.ones((m,n))*140

#IC
#first row should be zero 
#T[0,1:-1]=0

#BCs
#col 1 holds values at left side of the rod
#last col holds the other boundary 
T[:,0]=Tleft
T[:,-1]=Tright


A=np.diag((1+2*lam)*np.ones(n-2))+np.diag(-lam*np.ones(n-3),1)+\
  np.diag(-lam*np.ones(n-3),-1)
  
print(A)

#let's make the b for the first step and compare to our hand solution
#step 1 figure this out
#row 1 holds values of T(t=0)
#for l=1 we need T(l=0)
print(T[0,1:-1])
b=np.copy(T[0,1:-1])
b[0]+=lam*Tleft
b[-1]+=lam*Tright
print(b)

X=np.linalg.solve(A,b)

print("solution is "+str(X))

#step 2
#check we get the same result
T[1,1:-1]=X

print(T[:2])


#step3 extend for other steps in time
#we have already solved for l=1
#A remains the same 
#only b changes
for l in range(1,m-1):
	b=np.copy(T[l,1:-1])
	b[0]+=lam*Tleft
	b[-1]+=lam*Tright
	T[l+1,1:-1]=np.linalg.solve(A,b)

#print(T)

plt.figure()
label=[]
for i in range(int((tend/dt)+1)):
    plt.plot(x_val,T[i])
    label.append('t = '+str(i*dt) + ' s')

plt.xlabel("y")
plt.ylabel("T")
plt.title("T as a function of position")

plt.legend(label,bbox_to_anchor=(1.1, 1.05))