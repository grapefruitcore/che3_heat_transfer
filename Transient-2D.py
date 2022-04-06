# Group members: ABM Hussien, Evelyn Gondosiswanto, Michael Orrett, Yuan Chen
# Student ID: 20814321, 20820068, 20831409, 20822821

# Transient 2D Heat Transfer
# Cylindrical coordinates are used for this solution

# Boundary conditions for T(r,Φ,z,t)
# 1) T(r,Φ,0,t) = 165 degrees Celsius
# 2) T(r,Φ,L,t) = constant, finite value
#    --> dTdz = 0 @ z = L
# 3) T(0,Φ,z,t) = constant, finite value
#    --> dTdt = 0 @ r = 0
# 4) T(R,Φ,z,t) = Ta = 25 degrees Celsius
# 5) T(r,Φ,z,0) = Ta = 25 degrees Celsius

# Assumptions
# 1) Assume that the temperature does not vary significantly in the Φ direction compared to the r or z direction
#    --> only consider heat transfer in the r and z directions
# 2) Assume uniformity in the Φ direction
#    --> this allows us to solve for the entire cylinder by looking at the temperature profile of only a slice of it & assuming it's the same for the rest of the cylinder

import numpy as np
import numpy.linalg as lng
import matplotlib.pyplot as plt
import pandas as pd
import copy as copy

# constants
k = 0.145       # W/m degC
rou = 1435.9    # kg/m3
Cp = 1.57842    # W/kg
deltat = 50   # seconds
t_total = 1500  # total time
lamda = k*deltat/(rou*Cp)

# domain specifications
R = 0.15        # m
L = 0.25        # m
nR = 3          # number of unknown nodes in the R domain
nZ = 3          # number of unknown nodes in the Z domain
deltaR = R/nR   # spacing for the R nodes
deltaZ = L/nZ   # spacing for the Z nodes

# numerical boundary conditions
Ta = 25         # degrees Celsius
Tbot = 165      # degrees Celsius
TR = 25         # degrees Celsius

# building the coefficient matrix
A = np.zeros((nR*nZ,nR*nZ))
B = np.zeros((nR*nZ))
diag1 = np.diag((2*lamda)*(1/deltaR**2 + 1/deltaZ**2)*np.ones(len(np.diag(A))))

# building the equations one by one as things were complicated
diag1[0] = np.array([2*(1/deltaR**2 + 1/deltaZ**2), 2/deltaR**2, 0, 2/deltaZ**2, 0, 0, 0, 0, 0])
B[0] = 0

diag1[1] = np.array([1/deltaR**2, 2*(1/deltaR**2 + 1/deltaZ**2), 1/deltaR**2, 0, 2/deltaZ**2, 0, 0, 0, 0])
B[1] = 0

diag1[2] = np.array([0, 1/deltaR**2, 2*(1/deltaR**2 + 1/deltaZ**2), 0, 0, 2/deltaZ**2, 0, 0, 0])
B[2] = TR

diag1[3] = np.array([1/deltaZ**2, 0, 0, 2*(1/deltaR**2 + 1/deltaZ**2), 2/deltaR**2, 0, 1/deltaZ**2, 0, 0])
B[3] = 0

diag1[4] = np.array([0, 1/deltaZ**2, 0, 1/deltaR**2, 2*(1/deltaR**2 + 1/deltaZ**2), 1/deltaR**2, 0, 1/deltaZ**2, 0])
B[4] = 0

diag1[5] = np.array([0, 0, 1/deltaZ**2, 0, 1/deltaR**2, 2*(1/deltaR**2 + 1/deltaZ**2), 0, 0, 1/deltaZ**2])
B[5] = TR

diag1[6] = np.array([0, 0, 0, 1/deltaZ**2, 0, 0, 2*(1/deltaR**2 + 1/deltaZ**2), 2/deltaR**2, 0])
B[6] = Tbot

diag1[7] = np.array([0, 0, 0, 0, 1/deltaZ**2, 0, 1/deltaR**2, 2*(1/deltaR**2 + 1/deltaZ**2), 1/deltaR**2])
B[7] = Tbot

diag1[8] = np.array([0, 0, 0, 0, 0, 1/deltaZ**2, 0, 1/deltaR**2, 2*(1/deltaR**2 + 1/deltaZ**2)])
B[8] = Tbot + TR

diag1 = diag1*lamda

# initial T values at t = 0
T0 = np.ones(np.shape(B))*Ta
B = lamda*B
Bstart = copy.deepcopy(B)
B = B + T0

nt = t_total/deltat
final_T=[]
for j in range(int(nt)):
    final_T.append(T0)
final_T = np.array(final_T)

for i in range(int(nt-1)):
    b=np.copy(final_T[i,1:-1])
    b[0]+=lamda*Ta
    b[-1]+=lamda*TR
    temp = lng.solve(diag1,B)
    temp_vals = temp + final_T[i]
    final_T[i+1]=temp_vals

avg_temp = np.round(sum(list(final_T[-1]))/len(final_T[-1]),3)
final_T = np.round(np.array(final_T),3)
final_T_Pandas = pd.DataFrame(final_T)

print(f"These are the numbers for the temperature profile: \n{final_T_Pandas}")
print(f"The average temperature of the system is: {avg_temp} C")
  
final_Tplot = np.reshape(final_T[-1],(3,3))
plt.imshow(final_Tplot,extent=[0.15,0,0.25,0], cmap='hot_r', interpolation='nearest')
plt.xlabel("r (m)", fontsize=18)
plt.ylabel("z (m)", fontsize=18)
plt.title("The Temperature Profile of the Node System", fontsize=18)
plt.colorbar()
plt.show()

