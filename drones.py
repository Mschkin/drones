import cv2
import numpy as np
import time
import copy
import sys


# fluchtpunkt muss in die mitte verschoben werden


def splitinparts(frame, n):
    # print('splitin...',n)
    shape = np.shape(frame)[0:2]
    # matrix that contains normalized parts and the position of the upper left
    # corner as tuples
    return np.array([[normalise(frame[shape[0] // n * i: shape[0] // n *
                                      (i + 1) + shape[0] % n, shape[1] // n * j: shape[1] // n *
                                      (j + 1) + shape[1] % n]) for j in range(n)] for i in range(n)]),\
        np.array([[np.array([shape[0] // n * i, shape[1] // n * j])
                   for j in range(n)] for i in range(n)])


def normalise(frame):
    # checks that var is not 0
    return (frame - np.mean(frame, (0, 1))) / ((np.var(frame, (0, 1)) == 0) + np.var(frame, (0, 1)))**0.5


def scalarproduct(bigframe, smallframe, pos):
    bigframe = bigframe[searchradius + pos[0]:searchradius + pos[0] + np.shape(
        smallframe)[0], searchradius + pos[1]:searchradius + pos[1] + np.shape(smallframe)[1]]
    # select window
    bigframe = normalise(bigframe)
    return np.sum(bigframe * smallframe)


def scalarfieldcreator(frame1, frame2, partnumber):
    frame1 = frame1[searchradius:-searchradius, searchradius:-searchradius]
    partmatrix, pos = splitinparts(frame1, partnumber)
    scalarfield = np.zeros(
        (partnumber, partnumber, 2 * searchradius // searchstepsize, 2 * searchradius // searchstepsize))
    for x, vec in enumerate(partmatrix):
        for y, part in enumerate(vec):
            # print(np.shape(part))
            for i in range(-searchradius, searchradius, searchstepsize):
                for j in range(-searchradius, searchradius, searchstepsize):
                    scalarfield[x, y, (i + searchradius) // searchstepsize, (j + searchradius) // searchstepsize] =\
                        scalarproduct(
                            frame2, part, pos[x, y] + np.array([i, j]))
                    #visualiseweights(part,frame2,pos[x, y] +np.array([i, j]),scalarfield[x, y, (i + searchradius) // searchstepsize, (j + searchradius) // searchstepsize],frame1,x,y,partnumber)
    return scalarfield, pos, np.shape(partmatrix[0, 0])[:2]


def find(scalarfield, pos, smallframeshape):
    # print(scalarfield)
    # needs to be finished
    m = np.zeros((2, 2))
    qn = np.array([0., 0.])

    potential = np.zeros((72, 128))
    for i1, vecmat in enumerate(scalarfield):
        for i2, weights in enumerate(vecmat):
            weights = normalise(weights)

            nis = np.array([[[-j * searchstepsize + searchradius, i * searchstepsize - searchradius] for i in range(
                2 * searchradius // searchstepsize)] for j in range(2 * searchradius // searchstepsize)], dtype=float)
            cis = np.array([[ni@pos[i1, i2] for ni in vec]
                            for vec in nis], dtype=float)
            # remove weights bellow zero
            weights = weights * (weights > 0)
            potential += addpotential(nis, cis, weights)
            for x, vec in enumerate(nis):
                for y, ni in enumerate(vec):
                    m += weights[x, y] * np.tensordot(ni, ni, 0)
                    qn += weights[x, y] * ni * cis[x, y]

    allweights = np.array([w for k in weights for w in k])
    print("shape of allweights")
    plotpotential(allweights)
    print(m)
    print(np.linalg.inv(m))
    print(qn)
    print((2 * np.linalg.inv(m)@qn +
           smallframeshape + 2 * [2 * searchradius]) / 2)
    return (2 * np.linalg.inv(m)@qn + smallframeshape + 2 * [2 * searchradius]) / 2


def addpotential(nis, cis, weights):
    potential = np.zeros((72, 128))
    for index, wei in np.ndenumerate(weights):
        for pos, val in np.ndenumerate(potential):
            potential[pos] += wei * \
                (nis[index]@pos - cis[index])**2
    return potential


def plotpotential(pot):
    print("shape of plot ", np.shape(pot))
    pot = 255 - ((pot - np.min(pot)) /
                 (np.max(pot) - np.min(pot)) * 255 + 0.5).astype(np.uint8, casting='unsafe')

    print("middle: ", pot[36, 64],
          " lower right corner ", pot[71, 127])
    cv2.imshow('', pot)
    cv2.waitKey(5000)


def drawepipolarlines(pos, vpoint, f1, f2, smallframeshape):
    pos = [(2 * p + smallframeshape + searchradius * 2) /
           2 for vec in pos for p in vec]
    for p in pos:
        #print(p, vpoint)
        cv2.line(f1, (int(p[1] + 0.5), int(p[0] + 0.5)),
                 (int(vpoint[1] + 0.5), int(vpoint[0] + 0.5)), (255, 0, 0), 1)
        cv2.line(f2, (int(p[1] + 0.5), int(p[0] + 0.5)),
                 (int(vpoint[1] + 0.5), int(vpoint[0] + 0.5)), (255, 0, 0), 1)


def visualiseweights(part, f2, pos, weight, f1, i, j, n):
    print(pos)
    print(i, j)
    print(np.shape(part))
    print(n)
    shape = (np.shape(f1)[0], np.shape(f1)[1])
    f0 = np.zeros(np.shape(f2), dtype=np.uint8)
    f0[searchradius + pos[0]:pos[0] + np.shape(part)[0] + searchradius, searchradius + pos[1]:searchradius + pos[1] + np.shape(part)[1]] = f1[shape[0] // n * i: shape[0] // n *
                                                                                                                                              (i + 1) + shape[0] % n, shape[1] // n * j: shape[1] // n *
                                                                                                                                              (j + 1) + shape[1] % n]
    print(f0)
    cv2.imshow(str(weight), f2)
    cv2.waitKey(200)
    cv2.imshow(str(weight), f0)
    cv2.waitKey(200)
    cv2.imshow(str(weight), f0 // 2 + f2 // 2)
    cv2.waitKey(200)
    cv2.destroyAllWindows()


cap = cv2.VideoCapture('flasche.mp4')
searchradius = 8
searchstepsize = 2
#_, f1 = cap.read()
# print(np.shape(f1))
for k in range(10):
    _, f1 = cap.read()
_, f2 = cap.read()

cv2.resize(f1, (72, 128))
cv2.resize(f2, (72, 128))
print(np.shape(f1))
cf = copy.deepcopy(f1)

scalrfield, pos, smallframeshape = scalarfieldcreator(cf, f2, 9)
vpoint = find(scalrfield, pos, smallframeshape)
drawepipolarlines(pos, vpoint, f1, f2, smallframeshape)

while True:
    cv2.imshow('', f1)
    cv2.waitKey(500)
    cv2.imshow('', f2)
    cv2.waitKey(500)

""""  
#print(splitinparts(f1, 3))

f1 = f1[50:-50, 50:-50]
frames = splitinparts(f1, 3)
print(frames[0, 0])
print(np.shape(frames[0, 0]))
x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(x)
print(np.mean(frames, (2, 3)))
print(np.var(frames, (2, 3)))
#print((frames - np.mean(frames, (2, 3))) / np.var(frames, (2, 3)))
frames = np.array([[(i - np.mean(i, (0, 1))) / np.var(i, (0, 1))
                    ** 0.5 for i in vec] for vec in frames])
f2 = f2 - np.mean(f2)
print(frames)
print(np.shape(frames))

while True:

    cv2.imshow('', f2)
    cv2.waitKey(500)

print(scalarproduct(np.array([[1, 2], [0, 1]]),
                    np.array([[1, 1], [5, 1]]), [0, 0]), 'asfd')


"""
