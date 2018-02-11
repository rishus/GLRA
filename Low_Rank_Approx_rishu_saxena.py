# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:06:28 2015

@author: rishu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 4 06:46:15 2015

@author: rishu
"""

import numpy as np
import os
import matplotlib.image as mpimg
import re
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
import time

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    

def bar_char_time(propTime, svdTime):
    N = 3
    ind = np.arange(N)  # the x locations for the groups
    width = 0.3       # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, propTime, width, color='r')
    rects2 = ax.bar(ind+width, svdTime, width, color='y')
    ax.set_ylabel('Time (in seconds)')
    ax.set_title('')
    ax.set_xticks(ind+width)
    ax.legend( (rects1[0], rects2[0]), ('proposed Algorithm', 'SVD') )	
    ax.set_xticklabels( ('Face', 'Leaves', 'Menu') )
    plt.savefig("time.png")
    plt.show()
    
    return plt

def bar_char_prf(precision, recall, f1):
    N = 3
    ind = np.arange(N)  # the x locations for the groups
    width = 0.15      # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, precision, width, color='r')
    rects2 = ax.bar(ind+width, recall, width, color='y')
    rects3 = ax.bar(ind+(2 * width), f1, width, color='b')
    
    ax.set_ylabel('Time (in seconds)')
    ax.set_title('')
    ax.set_xticks(ind+width)
    ax.legend( (rects1[0], rects2[0], rects3[0]), ('Face', 'Leaves', 'Menu Match') )	
    ax.set_xticklabels( ('Precision', 'Recall', 'Fstat') )
    plt.savefig("pRF.png")
    plt.show()
    
    return plt

def bar_char_prf_thisvssvd(thisAlg, svdAlg):
    N = 3
    ind = np.arange(N)  # the x locations for the groups
    width = 0.15      # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, thisAlg, width, color='r')
    rects2 = ax.bar(ind+width, svdAlg, width, color='y')
    
    ax.set_ylabel('Time (in seconds)')
    ax.set_title('')
    ax.set_xticks(ind+width)
    ax.legend( (rects1[0], rects2[0]), ('proposed algorithm', 'svd') )	
    ax.set_xticklabels( ('Precision', 'Recall', 'Fstat') )
    plt.savefig("pRF_thisvsSvd.png")
    plt.show()
    
    return plt
    
########################################################         
                            # face dataset 
n = 400
r = 112
c = 92
A = np.zeros((r,c,n))
target = np.zeros((n))
for f in range(0,40):
    path = "/home/rishu/ImMLproject1/att_faces/s" + str(f+1)
    os.chdir(path)
    for i in range(0, 10):
        fn = str(i+1)+".pgm"
        A[:, :, 10*f + i] = read_pgm(fn, byteorder='<')
        target[10*f + i] = (f+1)
        
# SVD matrix
Asvd = np.zeros((r*c,n))
tmp = np.zeros((r, c))
for f in range(0,40):
    path = "/home/rishu/ImMLproject1/att_faces/s" + str(f+1)
    os.chdir(path)
    for i in range(0, 10):
        fn = str(i+1)+".pgm"
        tmp[:, :] = read_pgm(fn, byteorder='<')
        s = 0
        for k in range(0,c):
            for j in range(0,r):
                Asvd[s*r + j, 10*f + i] = tmp[j, k]
            s += 1
    
#    print f
#    pyplot.imshow(A[:, :, 10*f+ 1], pyplot.cm.gray)
#    pyplot.show()
###############################################

########################################################         
                            # leaf dataset 
path = "/home/rishu/ImMLproject1/leaves"
os.chdir(path)
img = mpimg.imread("image_0001.jpg")
gray = rgb2gray(img)
(r,c) = gray.shape
n = 186
A = np.zeros((r,c,n))
target = np.zeros((n))
for i in range(0, n):
    if i < 9:
        fn = "image_000"+str(i+1)+".jpg"
    elif (i >=9 and i < 99):
        fn = "image_00"+str(i+1)+".jpg"
    else:
        fn = "image_0"+str(i+1)+".jpg"

    img = mpimg.imread(fn)
    gray = rgb2gray(img)
    A[:, :, i] = np.array(gray)
    if i < 67:
        target[i] = 1
    elif (i >= 67 and i < 126):
        target[i] = 2
    else:
        target[i] = 3

# SVD approximation
Asvd = np.zeros((r*c,n))
for i in range(0, n):
    if i < 9:
        fn = "image_000"+str(i+1)+".jpg"
    elif (i >=9 and i < 99):
        fn = "image_00"+str(i+1)+".jpg"
    else:
        fn = "image_0"+str(i+1)+".jpg"
        
    img = mpimg.imread(fn)
    gray = rgb2gray(img)
    tmp = np.array(gray)
    s = 0
    for k in range(0,c):
        for j in range(0,r):
            Asvd[s*r + j, i] = tmp[j, k]
        s += 1

#    plt.imshow(gray, cmap = plt.get_cmap('gray'))
###############################################

########################################################         
                            # menu dataset 
path = "/home/rishu/ImMLproject1/menuMatch/foodimages"
os.chdir(path)
img = mpimg.imread("img1.jpg")  # note that images have different sizes here. i'm cropping to smallest
gray = rgb2gray(img)
#(r,c) = gray.shape  # to avoid memory error
cuisine = ['italian','soup', 'asian',]
n = 649
r=512
c=512
A = np.zeros((r,c,n))
target = np.zeros((n))
n_curr = 0
for cuis in cuisine:
    path = "/home/rishu/ImMLproject1/menuMatch/foodimages"
    os.chdir(path)
    onlyfiles = [ f for f in os.listdir(cuis) if os.path.isfile(os.path.join(cuis,f)) ]
    path = "/home/rishu/ImMLproject1/menuMatch/foodimages/"+cuis
    os.chdir(path)
    for fn in onlyfiles:
        img = mpimg.imread(fn)
        gray = rgb2gray(img)
        lr=330
        ur=330+r
        lc=330
        uc =330+c
        if(gray.shape[0] < ur):
            lr = 0
            ur = min(gray.shape[0], r)
        if(gray.shape[1] < uc):
            lc = 0
            uc = min(gray.shape[1], c)
        x = np.array(gray[lr:ur, lc:uc])
        A[0:ur-lr+1, 0:uc-lc+1, n_curr] = x
        if cuis == 'italian' :
            target[n_curr] = 1
        elif (cuis == 'soup'):
            target[n_curr] = 2
        else:
            target[n_curr] = 3
        n_curr += 1

# SVD approximation
Asvd = np.zeros((r*c,n))
for i in range(0, n):
    path = "/home/rishu/ImMLproject1/menuMatch/foodimages"
    os.chdir(path)
    onlyfiles = [ f for f in os.listdir(cuis) if os.path.isfile(os.path.join(cuis,f)) ]
    path = "/home/rishu/ImMLproject1/menuMatch/foodimages/"+cuis
    os.chdir(path)
        
    for fn in onlyfiles:
        img = mpimg.imread(fn)
        gray = rgb2gray(img)
        lr=330
        ur=330+r
        lc=330
        uc =330+c
        if(gray.shape[0] < ur):
            lr = 0
            ur = min(gray.shape[0], r)
        if(gray.shape[1] < uc):
            lc = 0
            uc = min(gray.shape[1], c)
        
        x = np.array(gray[lr:ur, lc:uc])
        s = 0
        for k in range(0,c):
            for j in range(0,r):
                Asvd[s*r + j, i] = x[j, k]
            s += 1

#    plt.imshow(gray, cmap = plt.get_cmap('gray'))

    
###############################################

path = "/home/rishu/ImMLproject1/"
os.chdir(path)    

dall = [2*i for i in range(0,10) ]
thisAcc = []
svdAcc = []
d = 20 #dall[0]
l1 = d
l2 = d
L = np.zeros((r,l1))
L[0:l1, 0:l1] = np.eye(l1)
tol = 0.001
err = 10
i = 1
t0 = time.time()
while (err > tol):
    Mright = np.zeros((c, c))
    for j in range(0, n):
        AtL = np.dot(np.transpose(A[:, :, j]), L)
        Mright = Mright + np.dot(AtL, np.transpose(AtL))
    
    lamdaRight, phiRight = np.linalg.eig(Mright)
    Rnew = phiRight[:, 0:l2]
    
    Mleft = np.zeros((r, r))
    for j in range(0, n):
        AR =  np.dot(A[:, :, j], Rnew)
        Mleft = Mleft + np.dot(AR, np.transpose(AR))
    
    lamdaLeft, phiLeft = np.linalg.eig(Mleft)
    Lnew = phiLeft[:, 0:l1]
    err = np.linalg.norm(Lnew - L)
    print i, ', ', err
    L = Lnew
    i += 1

L = Lnew
R = Rnew
D = np.zeros((l1, l2, n))
for j in range(0, n):
    D[:, :, j] = np.dot(np.transpose(L),np.dot(A[:, :, j], R))

t1 = time.time()
time_prop_alg = t1-t0

for i in range(0, n):
    err = err + np.linalg.norm(A[:, :, i] - np.dot(L, np.dot(D[:,:,j], np.transpose(R))))

# reconstruction
Arec = np.zeros((r,c,n))
for j in range(0,n):
    Arec[:,:,j] = np.dot(L, np.dot(D[:,:,j], np.transpose(R)))


#### SVD ####
t2 = time.time()
U, S, V = np.linalg.svd(Asvd)
t3 = time.time()
time_svd = t3-t2
Ufinal = U[:,0:l1]
Vfinal = V[:,0:l1]
sfinal = S[0:l1]
Sdiag = np.zeros((r*c, n))
for i in range(0, len(sfinal)):
    Sdiag[i,i] = sfinal[i]

B = np.dot(U,np.dot(Sdiag, V)) 

svdrec = np.zeros((r,c,n))
for k in range(0, n):
    s = 0
    for j in range(0,c):
        for i in range(0,r):
            svdrec[i, j, k] = B[r*s +  i, k]  
        s += 1

# classification: using k-NN
Fthis = np.zeros((n, r*c))      
#Fthis = np.zeros((int(n/4), r*c))  # for running cross validation on the menu dataset
Fsvd = np.zeros((n, r*c))
#target_reduced = np.zeros((int(n/4)))  # for running cross validation on the menu dataset
for k in range(0, len(target)):
    beg = 0
    end = c
    for i in range(0,r):
        Fthis[k][ beg : end] = [Arec[i, j , k] for j in range(0,c)]
        Fsvd[k][ beg : end] = [svdrec[i, j ,k] for j in range(0,c)]
        beg = end
        end = end + c
#    target_reduced[k] = target[4*k]
        
this_fnum = []
svd_fnum = []
x = [1] #, 2, 3]
for k in x:
    clf_knn = KNeighborsClassifier(n_neighbors=k)
    this_scores_knn = cross_validation.cross_val_score(clf_knn, Fthis, target, cv = 10, scoring = 'accuracy')
#    svd_scores_knn = cross_validation.cross_val_score(clf_knn, Fsvd, target, cv = 10, scoring = 'f1')
    this_fnum.append( sum(this_scores_knn)/10)
#    svd_fnum.append( sum(svd_scores_knn)/10)

    
plt.figure()
plt.plot(x, this_fnum, label='proposed'), plt.hold('True')
plt.plot(x, svd_fnum, label='SVD')
plt.legend() #(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.ylabel('accuracy')
plt.xlabel('K-Nearest Neighbors')
plt.savefig("face_accuracy.png")
plt.hold(False)


numrows = 2
numcols = 4
fig = plt.figure()
ax1 = fig.add_subplot(numrows,numcols,1)
ax1.imshow(A[:, :, 1], plt.cm.gray)
ax1.set_xticks([]) 
ax1.set_yticks([]) 
#plt.show()
ax2 = fig.add_subplot(numrows,numcols,numcols+1)
ax2.imshow(Arec[:, :, 1], plt.cm.gray)
ax2.set_xticks([]) 
ax2.set_yticks([]) 
#plt.show()
ax3 = fig.add_subplot(numrows,numcols,2*numcols+1)
ax3.imshow(svdrec[:, :, 1], plt.cm.gray)
ax3.set_xticks([]) 
ax3.set_yticks([]) 
#plt.show()

ax4 = fig.add_subplot(numrows,numcols,2)
ax4.imshow(A[:, :, 50], plt.cm.gray)
ax4.set_xticks([]) 
ax4.set_yticks([]) 
#plt.show()
ax5 = fig.add_subplot(numrows,numcols,numcols+2)
ax5.imshow(Arec[:, :, 50], plt.cm.gray)
ax5.set_xticks([]) 
ax5.set_yticks([]) 
#plt.show()
ax6 = fig.add_subplot(numrows,numcols,2*numcols + 2)
ax6.imshow(svdrec[:, :, 11], plt.cm.gray)
ax6.set_xticks([]) 
ax6.set_yticks([]) 

ax7 = fig.add_subplot(numrows,numcols,3)
ax7.imshow(A[:, :, 200], plt.cm.gray)
ax7.set_xticks([]) 
ax7.set_yticks([]) 
#plt.show()
ax8 = fig.add_subplot(numrows,numcols,numcols+3)
ax8.imshow(Arec[:, :, 200], plt.cm.gray)
ax8.set_xticks([]) 
ax8.set_yticks([]) 
##plt.show()
ax9 = fig.add_subplot(numrows,numcols,2 * numcols + 3)
ax9.imshow(svdrec[:, :, 21], plt.cm.gray)
ax9.set_xticks([]) 
ax9.set_yticks([]) 

ax10 = fig.add_subplot(numrows,numcols, 4)
ax10.imshow(A[:, :, 600], plt.cm.gray)
ax10.set_xticks([]) 
ax10.set_yticks([]) 
##plt.show()
ax11 = fig.add_subplot(numrows,numcols,numcols + 4)
ax11.imshow(Arec[:, :, 600], plt.cm.gray)
ax11.set_xticks([]) 
ax11.set_yticks([]) 
#plt.show()
ax12 = fig.add_subplot(numrows,numcols,2*numcols + 4)
ax12.imshow(svdrec[:, :, 31], plt.cm.gray)
ax12.set_xticks([]) 
ax12.set_yticks([]) 

ax13 = fig.add_subplot(numrows,numcols,5)
ax13.imshow(A[:, :, 41], plt.cm.gray)
ax13.set_xticks([]) 
ax13.set_yticks([]) 
#plt.show()
ax14 = fig.add_subplot(numrows,numcols, numcols + 5)
ax14.imshow(Arec[:, :, 41], plt.cm.gray)
ax14.set_xticks([]) 
ax14.set_yticks([]) 
plt.show()
ax15 = fig.add_subplot(numrows,numcols,2*numcols +5)
ax15.imshow(svdrec[:, :, 41], plt.cm.gray)
ax15.set_xticks([]) 
ax15.set_yticks([]) 


fig.show()
fig.savefig("food.png")


# plotting time
propTime = [0.64741, 77.50, 40.0479]
svdTime = [33.8146, 130.00, 130.00]
bar_char_time(propTime, svdTime)


# plotting PRF
numrows = 1
numcols = 2
fig = plt.figure()
ax1 = fig.add_subplot(numrows,numcols,1)
ax1.imshow()
ax1.set_xticks([]) 
ax1.set_yticks([]) 
ax2 = fig.add_subplot(numrows,numcols,numcols+1)
ax2.imshow(Arec[:, :, 1], plt.cm.gray)
ax2.set_xticks([]) 
ax2.set_yticks([]) 

precision = [0.9737, 0.722, 0.5975]
recall = [0.982, 0.7029, 0.56213]
f1 = [0.97, 0.7, 0.5448]
bar_char_prf(precision, recall, f1)

thisprf = [0.9737, 0.982, 0.97]
svdprf = [0.006, 0.025, 0.001]
bar_char_prf_thisvssvd(thisprf, svdprf)


# plotting effect of d on accuracy
dval = [5, 10, 15, 20]
face = [0.982499, 0.982499, 0.97999,  0.98249]
leaves = [0.62836, 0.7137, 0.708479, 0.702923976608]
food = [0.554779411, 0.5676470588, 0.580147058823, 0.702923976]
plt.figure()
plt.plot(dval, face, 'k-', linewidth=3,label='Face'), plt.hold('True')
plt.plot(dval, leaves, 'r-x', linewidth=3, label='Leaves')
plt.plot(dval, food, 'b--', linewidth=3, label='Menu Match')
plt.legend() #(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.ylabel('accuracy')
plt.xlabel('d')
plt.savefig("efeectOfd.png")
plt.hold(False)

