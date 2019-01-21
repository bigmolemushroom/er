from matplotlib import pyplot as plt
import numpy as np

SAMPLE_NUM = 100

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
        A = A + np.transpose(A)
        np.fill_diagonal(A, 3)
        A = A < 2*prob

        if(isConn(A) and degreeAtLeast(A, minDeg)):
            return A

#
def ERBoundScaleWithD(nodeNum, prob, dRange, fileName):
    #
    WList = [None for i in range(dRange[1]+1)]
    eigVals = []
    bounds = []
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
            if(sam == 0):
                WList[d] = W
            else:
                WList[d] += W
    
    for d in range(2, dRange[1]+1):
        WList[d] = WList[d] / SAMPLE_NUM
        lambda2 = np.sort(np.linalg.eigvals(WList[d]))[nodeNum-2]
        eigVals.append(lambda2)
        bounds.append(-1/np.log(lambda2))

     # plot the second largest eigenvalues of W, e.g., lambda_2(W)
    plt.figure(1)
    plt.plot(list(range(dRange[0],dRange[1]+1)), eigVals)
    #plt.plot(list(range(dRange[0],dRange[1]+1)), eigVals, list(range(dRange[0],dRange[1]+1)), bounds)
    plt.title('n=' + str(nodeNum) + ', d=[' + str(dRange[0]) + ':' + str(dRange[1]) + ']')
    plt.ylabel('Second lagest eigenvalue')
    plt.xlabel('size of group exchange (d)')
    plt.savefig(fileName[0], format='eps', dpi=1000)
    
    
    # plot the value -1/(log(lambda_2(W))) which is proportional to the bounds on the averaging time (for any given epsilon)
    plt.figure(2)
    plt.plot(list(range(dRange[0],dRange[1]+1)), bounds)
    plt.title('n=' + str(nodeNum) + ', d=[' + str(dRange[0]) + ':' + str(dRange[1]) + ']')
    plt.ylabel('Bound')
    plt.xlabel('size of group exchange (d)')
    plt.savefig(fileName[1], format='eps', dpi=1000)
    plt.show()
    



def main():
    ERBoundScaleWithD(nodeNum = 100, prob = 0.3, dRange = [2,50], fileName=['ERFigure3-1.eps', 'ERFigure3-2.eps'])


if(__name__ == '__main__'):
    main()
