from matplotlib import pyplot as plt
import numpy as np

SAMPLE_NUM = 50

# Check if the graph corresponding to the adjacency matrix is connected
def isConn(adjMat):
    visited = set()
    stack = [0]
    while(stack):
        v = stack.pop()
        if(v not in visited):
            visited.add(v)
            for i in range(len(adjMat)):
                if(adjMat[v][i] and (i not in visited)):
                    stack.append(i)
    if(len(visited) == len(adjMat)):
        return True
    else:
        return False

def degreeAtLeast(A, minDeg):
    deg = np.sum(A, axis=1)
    if(np.min(deg) < minDeg):
        return False
    else:
        return True

# Generate the adjacency of a connected ER random graph
def genConnAdjMat(nodeNum, prob, minDeg):
    # adjacency matrix
    while(True):
        A = np.random.rand(nodeNum, nodeNum)
        A = np.triu(A)
        np.fill_diagonal(A, 0)
        A = A > 1-prob
        A += A.T

        if(isConn(A) and degreeAtLeast(A, minDeg)):
            return A

#
def ERBoundScaleWithD(nodeNum, prob, dRange, fileName):
    #
    WList = [None for i in range(dRange[1]+1)]
    eigVals = [None for i in range(dRange[1]+1)]
    bounds = [None for i in range(dRange[1]+1)]
    boundsInv = [None for i in range(dRange[1]+1)]
    ratio  = [None for i in range(dRange[1]+1)]
    for sam in range(SAMPLE_NUM):
        # print progress

        # adjacency matrix
        A = genConnAdjMat(nodeNum, prob, 2)
        #A = np.array([[0, 1, 1, 1, 1],
        #              [1, 0, 1, 1, 1],
        #              [1, 1, 0, 1, 0],
        #              [1, 1, 1, 0, 0],
        #              [1, 1, 0, 0, 0]])
        deg = np.sum(A, axis = 1)

        # W
        for d in range(dRange[0], dRange[1]+1):
            print('Sampling ER graphs', sam+1, '/', SAMPLE_NUM, ', d = ', d, '/', dRange[1], '\r', end='')
            # true size of group information exchange
            D = np.zeros(nodeNum)
            D1 = np.zeros(nodeNum)
            D2 = np.zeros(nodeNum)
            for i in range(nodeNum):
                D[i] = min(d, deg[i])
                D1[i] = (D[i]-1)*(D[i]-2)/D[i]/deg[i]/(deg[i]-1)
                D2[i] = (D[i]-1)*(D[i]-1)/D[i]/deg[i]

            #
            E = np.zeros((nodeNum, nodeNum))
            for i in range(nodeNum):
                for j in range(i+1, nodeNum):
                    for l in range(nodeNum):
                        if(i != j and A[i][l] and A[j][l]):
                            E[i][j] += D1[l]
            E += E.T

            #
            W = np.zeros((nodeNum, nodeNum))
            for i in range(nodeNum):
                for j in range(i+1, nodeNum):
                    W[i][j] += 1 / nodeNum * ((D[i]-1)/D[i]/deg[i]*A[i][j] + (D[j]-1)/D[j]/deg[j]*A[i][j] + E[i][j])
            W += W.T
            #
            F = np.zeros(nodeNum)
            for i in range(nodeNum):
                for l in range(nodeNum):
                    if(A[i][l]):
                        F[i] += D2[l]
            for i in range(nodeNum):
                W[i][i] = 1 - 1 / nodeNum * ((D[i]-1)/D[i] + F[i])

            # record W
            lambda2 = np.sort(np.linalg.eigvals(W))[nodeNum-2]

            if(sam == 0):
                eigVals[d] = (lambda2)
                bounds[d] = (- 1 / np.log(lambda2))
            else:
                eigVals[d] += lambda2
                bounds[d] += - 1 / np.log(lambda2)
    
    for d in range(2, dRange[1]+1):
        eigVals[d] /= SAMPLE_NUM
        bounds[d] /= SAMPLE_NUM
        boundsInv[d] = 1/bounds[d]
        ratio[d] = (bounds[d] * (d-1))

     # plot the second largest eigenvalues of W, e.g., lambda_2(W)
    plt.figure(1)
    plt.plot(list(range(dRange[0],dRange[1]+1)), eigVals[dRange[0]:])
    #plt.plot(list(range(dRange[0],dRange[1]+1)), eigVals, list(range(dRange[0],dRange[1]+1)), bounds)
    plt.title('n=' + str(nodeNum) + ', p=' + str(prob) + ', d=[' + str(dRange[0]) + ':' + str(dRange[1]) + ']')
    plt.ylabel('second lagest eigenvalue')
    plt.xlabel('size of group exchange (d)')
    plt.savefig(fileName+'-1.eps', format='eps', dpi=1000)
    
    
    # plot the value -1/(log(lambda_2(W))) which is proportional to the bounds on the averaging time (for any given epsilon)
    plt.figure(2)
    plt.plot(list(range(dRange[0],dRange[1]+1)), bounds[dRange[0]:])
    plt.title('n=' + str(nodeNum) + ', p=' + str(prob) + ', d=[' + str(dRange[0]) + ':' + str(dRange[1]) + ']')
    plt.ylabel('bound')
    plt.xlabel('size of group exchange (d)')
    plt.savefig(fileName+'-2.eps', format='eps', dpi=1000)

    plt.figure(3)
    plt.plot(list(range(dRange[0],dRange[1]+1)), boundsInv[dRange[0]:])
    plt.title('n=' + str(nodeNum) + ', p=' + str(prob) + ', d=[' + str(dRange[0]) + ':' + str(dRange[1]) + ']')
    plt.ylabel('inverse of bound')
    plt.xlabel('size of group exchange (d)')
    plt.savefig(fileName+'-3.eps', format='eps', dpi=1000)

    plt.figure(4)
    plt.plot(list(range(dRange[0],dRange[1]+1)), ratio[dRange[0]:])
    plt.ylim(bottom = 0)
    plt.title('n=' + str(nodeNum) + ', p=' + str(prob) + ', d=[' + str(dRange[0]) + ':' + str(dRange[1]) + ']')
    plt.ylabel('(d-1) * bound')
    plt.xlabel('size of group exchange (d)')
    plt.savefig(fileName+'-4.eps', format='eps', dpi=1000)
    plt.show()
    
def ERBoundScaleWithN(nRange, prob, d, fileName):
    #
    nList = []
    for i in range((nRange[1]-nRange[0])//nRange[2]+1):
        nList.append(nRange[0]+i*nRange[2])
    #
    WList = [None for i in range((nRange[1]-nRange[0])//nRange[2]+1)]
    bounds = []
    for n in range((nRange[1]-nRange[0])//nRange[2]+1):
        nodeNum = nRange[0]+n*nRange[2]
        #
        for sam in range(SAMPLE_NUM):
            # print progress
            print('n=', n, '/', (nRange[1]-nRange[0])//nRange[2]+1, ', sam=', sam, '/', SAMPLE_NUM, '\r', end = '')

            # adjacency matrix
            A = genConnAdjMat(nodeNum, prob, 2)
            deg = np.sum(A, axis = 1)

            # W
            D = np.zeros(nodeNum)
            D1 = np.zeros(nodeNum)
            D2 = np.zeros(nodeNum)
            for i in range(nodeNum):
                D[i] = min(d, deg[i])
                D1[i] = (D[i]-1)*(D[i]-2)/D[i]/deg[i]/(deg[i]-1)
                D2[i] = (D[i]-1)*(D[i]-1)/D[i]/deg[i]
            #
            E = np.zeros((nodeNum, nodeNum))
            for i in range(nodeNum):
                for j in range(i+1, nodeNum):
                    for l in range(nodeNum):
                        if(i != j and A[i][l] and A[j][l]):
                            E[i][j] += D1[l]
            E += E.T

            #
            F = np.zeros(nodeNum)
            for i in range(nodeNum):
                for l in range(nodeNum):
                    if(A[i][l]):
                        F[i] += D2[l]

            #
            W = np.zeros((nodeNum, nodeNum))
            for i in range(nodeNum):
                for j in range(i+1, nodeNum):
                    W[i][j] += 1 / nodeNum * ((D[i]-1)/D[i]/deg[i]*A[i][j] + (D[j]-1)/D[j]/deg[j]*A[i][j] + E[i][j])
            W += W.T

            for i in range(nodeNum):
                W[i][i] = 1 - 1 / nodeNum * ((D[i]-1)/D[i] + F[i])

            if(sam == 0):
                WList[n] = W
            else:
                WList[n] += W
            
    for n in range((nRange[1]-nRange[0])//nRange[2]+1):
        nodeNum = nRange[0]+n*nRange[2]
        WList[n] = WList[n] / SAMPLE_NUM
        lambda2 = np.sort(np.linalg.eigvals(WList[n]))[nodeNum-2]
        bounds.append(-1/np.log(lambda2))
    
    
    # plot the value -1/(log(lambda_2(W))) which is proportional to the bounds on the averaging time (for any given epsilon)
    plt.figure(1)
    plt.plot(nList, bounds)
    plt.title('n=[' + str(nRange[0]) + ':' + str(nRange[1]) + ']' + ', p=' + str(prob) + ', d=' + str(d))
    plt.ylabel('Bound')
    plt.xlabel('n')
    plt.savefig(fileName, format='eps', dpi=1000)
    plt.show()

def ERBoundScaleWithNSameK(nRange, k, d, fileName):
    #
    nList = []
    for i in range((nRange[1]-nRange[0])//nRange[2]+1):
        nList.append(nRange[0]+i*nRange[2])
    #
    WList = [None for i in range((nRange[1]-nRange[0])//nRange[2]+1)]
    bounds = []
    for n in range((nRange[1]-nRange[0])//nRange[2]+1):
        nodeNum = nRange[0]+n*nRange[2]
        #
        for sam in range(SAMPLE_NUM):
            # print progress
            print('n=', n, '/', (nRange[1]-nRange[0])//nRange[2]+1, ', sam=', sam, '/', SAMPLE_NUM, '\r', end = '')

            # adjacency matrix
            prob = k / (nodeNum-1)
            A = genConnAdjMat(nodeNum, prob, 2)
            deg = np.sum(A, axis = 1)

            # W
            D = np.zeros(nodeNum)
            D1 = np.zeros(nodeNum)
            D2 = np.zeros(nodeNum)
            for i in range(nodeNum):
                D[i] = min(d, deg[i])
                D1[i] = (D[i]-1)*(D[i]-2)/D[i]/deg[i]/(deg[i]-1)
                D2[i] = (D[i]-1)*(D[i]-1)/D[i]/deg[i]
            #
            E = np.zeros((nodeNum, nodeNum))
            for i in range(nodeNum):
                for j in range(i+1, nodeNum):
                    for l in range(nodeNum):
                        if(i != j and A[i][l] and A[j][l]):
                            E[i][j] += D1[l]
            E += E.T

            #
            F = np.zeros(nodeNum)
            for i in range(nodeNum):
                for l in range(nodeNum):
                    if(A[i][l]):
                        F[i] += D2[l]

            #
            W = np.zeros((nodeNum, nodeNum))
            for i in range(nodeNum):
                for j in range(i+1, nodeNum):
                    W[i][j] += 1 / nodeNum * ((D[i]-1)/D[i]/deg[i]*A[i][j] + (D[j]-1)/D[j]/deg[j]*A[i][j] + E[i][j])
            W += W.T

            for i in range(nodeNum):
                W[i][i] = 1 - 1 / nodeNum * ((D[i]-1)/D[i] + F[i])

            if(sam == 0):
                WList[n] = W
            else:
                WList[n] += W
            
    for n in range((nRange[1]-nRange[0])//nRange[2]+1):
        nodeNum = nRange[0]+n*nRange[2]
        WList[n] = WList[n] / SAMPLE_NUM
        lambda2 = np.sort(np.linalg.eigvals(WList[n]))[nodeNum-2]
        bounds.append(-1/np.log(lambda2))
    
    
    # plot the value -1/(log(lambda_2(W))) which is proportional to the bounds on the averaging time (for any given epsilon)
    plt.figure(1)
    plt.plot(nList, bounds)
    plt.title('n=[' + str(nRange[0]) + ':' + str(nRange[1]) + ']' + ', p=' + str(prob) + ', d=' + str(d))
    plt.ylabel('Bound')
    plt.xlabel('n')
    plt.savefig(fileName, format='eps', dpi=1000)
    plt.show()

def main():
    ERBoundScaleWithD(nodeNum = 100, prob = 0.4, dRange = [2,50], fileName='ERFigure03')
    #ERBoundScaleWithN(nRange = [20, 200, 20], prob = 1, d = 3, fileName = 'ERFigure9.eps')
    #ERBoundScaleWithNSameK(nRange = [20, 200, 20], k = 10, d = 3, fileName = 'ERFigure8.eps')


if(__name__ == '__main__'):
    main()
