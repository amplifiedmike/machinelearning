import numpy as np

def featureScaling(x,fs_type='mean normalization'):
    x_mu = np.mean(x)
    x_max = np.max(x)
    x_min = np.min(x)

    if(fs_type == 'mean normalization'):
        x_meanNormalization = (x-x_mu)/(x_max-x_min)
        return x_meanNormalization
    else:
        return x

def costFunction_lin(x,y,th0,th1,m):
    jsum = 0
    for elements in range(len(x)):
        jsum += (hypothesis(x[elements],th0,th1) - y[elements])**2
    j = jsum/(2*m)
    return j

def hypothesis(x,th0,th1):
    h = th0 + th1*x
    return h

def updateCoeffs_lin(x,y,th0,th1,alpha,m):
    th0_pd = 0
    th1_pd = 0
    for elements in range(len(x)):
        th0_pd += -1*(y[elements]-(th0+th1*x[elements]))
        th1_pd += -1*(x[elements]*(y[elements]-(th0+th1*x[elements])))
        #print("th0_pd: " + str(th0_pd) + ", th1_pd: " + str(th1_pd))
    th0_temp = th0 - alpha*(th0_pd/m)
    th1_temp = th1 - alpha*(th1_pd/m)

    return th0_temp, th1_temp

def costFunction2_lin(x1,x2,y,th0,th1,th2,m):
    jsum = 0
    for elements in range(len(x1)):
        jsum += (hypothesis2(x1[elements],x2[elements],th0,th1,th2) - y[elements])**2
    j = jsum/(2*m)
    return j

def hypothesis2(x1,x2,th0,th1,th2):
    h = th0 + th1*x1 + th2*x2
    return h

def updateCoeffs2_lin(x1,x2,y,th0,th1,th2,alpha,m):
    th0_pd = 0
    th1_pd = 0
    th2_pd = 0
    for elements in range(len(x1)):
        th0_pd += -1*(y[elements]-(th0+th1*x1[elements]+th2*x2[elements]))
        th1_pd += -1*(y[elements]-(th0+th1*x1[elements]+th2*x2[elements]))*x1[elements]
        th2_pd += -1*(y[elements]-(th0+th1*x1[elements]+th2*x2[elements]))*x2[elements]
        #print("th0_pd: " + str(th0_pd) + ", th1_pd: " + str(th1_pd))
    th0_temp = th0 - alpha*(th0_pd/m)
    th1_temp = th1 - alpha*(th1_pd/m)
    th2_temp = th2 - alpha*(th2_pd/m)

    return th0_temp, th1_temp, th2_temp