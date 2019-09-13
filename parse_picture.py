import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools as it

invert = True
find_contours = True

img = cv2.imread('img/batman.png',0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
#ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
#ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
#ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

if find_contours:
    thresh1_inv = cv2.bitwise_not(thresh1)
    edged = cv2.Canny(thresh1_inv, 30, 200)
    cv2.waitKey(0)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if invert:
        edged_inv = cv2.bitwise_not(edged)
    else:
        edged_inv = edged
    #cv2.imshow('Contour', edged_inv)
    #cv2.waitKey(0)
else:
    edged_inv = thresh1

coordinates = []
for i,j in it.product(*[range(a) for a in edged_inv.shape]):
    if edged_inv[i][j] == 0:
        coordinates.append((i,j))

ordered_coordinates = []
crnt_coordinate = coordinates[0]
nrst_coordinate = (0,0)
len_coordinates = len(coordinates)

print(len_coordinates)
for i in range(len_coordinates):
    crnt_dist = 0
    smlst_distance = 100000000
    for j in range(len_coordinates-i):
        if i == j: continue
        crnt_dist = (crnt_coordinate[0]-coordinates[j][0])**2 + (crnt_coordinate[1]-coordinates[j][1])**2

        if crnt_dist < smlst_distance:
            #print(crnt_dist)
            nrst_coordinate = coordinates[j]
            smlst_distance = crnt_dist
    #print(coordinates)
    coordinates.remove(nrst_coordinate)
    crnt_coordinate = nrst_coordinate
    ordered_coordinates.append(nrst_coordinate)

result = np.array([x - 1.j * y for y,x in ordered_coordinates])
np.savetxt('img/batman.txt', result)
