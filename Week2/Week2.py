import numpy as np
import matplotlib.pyplot as plt
import os
import gradientDescent as grd
from mpl_toolkits.mplot3d import Axes3D

#example 1
#extract data from files
script_dir = os.path.dirname(__file__)
f1 = open(script_dir+"/ex1data1.txt","r")
x1 =[]
y1=[]

for line in f1:
    currentline = line.split(",")
    x1.append(float(currentline[0]))
    y1.append(float(currentline[1]))

#Feature Scaling
x1_fs = grd.featureScaling(x1)

x1 = np.array(x1)
x1_fs = np.array(x1_fs)
y1 = np.array(y1)

#linear fit
th0 = []
th1 = []
j1=[]

alpha1 = 1
m1 = len(x1_fs)
iterations = 100

#first elements 
th0.append(0)
th1.append(0)
j1.append(grd.costFunction_lin(x1_fs,y1,th0[0],th1[0],m1))
#print( "th0: " + str( np.round(th0[0],3) ) + ", th1: " + str( np.round(th1[0],3) ) + ", j: " + str( np.round(j1[0],3) ) )

for i in range(1,iterations):
    #print(i)
    th0_temp, th1_temp = grd.updateCoeffs_lin(x1_fs,y1,th0[i-1],th1[i-1],alpha1,m1)
    th0.append(th0_temp)
    th1.append(th1_temp)
    j1.append(grd.costFunction_lin(x1_fs,y1,th0[i],th1[i],m1))
    #print( "th0: " + str( np.round(th0[i],3) ) + ", th1: " + str( np.round(th1[i],3) ) + ", j: " + str( np.round(j1[i],3) ) )

th0 = np.asarray(th0)
th1 = np.asarray(th1)

# print(th0[0])
# print(th1[0])
# print(th0[len(th0)-1])
# print(th1[len(th1)-1])

y1_first = th0[0] + th1[0]*x1_fs
y1_fit = th0[len(th0)-1] + th1[len(th1)-1]*x1_fs

#print(len(x1))
#print(len(x1_fs))
#print(len(y1))
#print(len(y1_fit))

plt.figure()
plt.scatter(x1_fs,y1)
plt.plot(x1_fs,y1_fit)
plt.plot(x1_fs,y1_first)
#plt.show()





#example 2
#extract data from files
f2 = open(script_dir+"/ex1data2.txt","r")
x2_1=[]
x2_2=[]
y2=[]

for line in f2:
    currentline = line.split(",")
    x2_1.append(float(currentline[0]))
    x2_2.append(float(currentline[1]))
    y2.append(float(currentline[2]))

#Feature Scaling
x2_1fs = grd.featureScaling(x2_1)
x2_2fs = grd.featureScaling(x2_2)

x2_1 = np.array(x2_1)
x2_1fs = np.array(x2_1fs)
x2_2fs = np.array(x2_2fs)
y2 = np.array(y2)

#linear fit
th0_2 = []
th1_2 = []
th2_2 = []
j2=[]

alpha2 = 0.5
m2 = len(x2_1fs)
iterations2 = 100

#first elements 
th0_2.append(0)
th1_2.append(0)
th2_2.append(0)
j2.append(grd.costFunction2_lin(x2_1fs, x2_2fs, y2, th0_2[0], th1_2[0], th2_2[0], m2))
#print( "th0: " + str( np.round(th0_2[0],3) ) + ", th1: " + str( np.round(th1_2[0],3) ) + ", th2: " + str( np.round(th2_2[0],3) ) + ", j: " + str( np.round(j2[0],3) ) )

for i in range(1,iterations2):
    #print(i)
    th0_2temp, th1_2temp, th2_2temp = grd.updateCoeffs2_lin(x2_1fs,x2_2fs,y2,th0_2[i-1],th1_2[i-1],th2_2[i-1],alpha2,m2)
    th0_2.append(th0_2temp)
    th1_2.append(th1_2temp)
    th2_2.append(th2_2temp)
    j2.append(grd.costFunction2_lin(x2_1fs, x2_2fs, y2, th0_2[i], th1_2[i], th2_2[i], m2))
    #print( "th0: " + str( np.round(th0[i],3) ) + ", th1: " + str( np.round(th1[i],3) ) + ", j: " + str( np.round(j1[i],3) ) )

th0_2 = np.asarray(th0_2)
th1_2 = np.asarray(th1_2)
th2_2 = np.asarray(th2_2)

index = len(th0_2)-1
y2_first = th0_2[0] + th1_2[0]*x2_1fs + th2_2[0]*x2_2fs
y2_fit = th0_2[index] + th1_2[index]*x2_1fs +th2_2[index]*x2_2fs
x2_test1 = np.linspace(-1.5,1.5,num=100)
x2_test2 = np.linspace(-1.5,1.5,num=100)
x2_test1, x2_test2 = np.meshgrid(x2_test1,x2_test2)
y2_fit2 = th0_2[index] + th1_2[index]*x2_test1 +th2_2[index]*x2_test2
print(x2_test1)


fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot_surface(x2_test1,x2_test2,y2_fit2)
ax.scatter(x2_1fs,x2_2fs,y2)
#Axes3D.scatter(x2_1fs,x2_2fs,y2)
#plt.plot(x2_1fs,y2_fit)
#plt.plot(x2_1fs,y2_first)
#plt.plot(x2_2fs,y2_fit)
#plt.plot(x2_2fs,y2_first)
plt.show()
