import numpy as np

def produceSymmMatrix(n):
    # create random invertible symmetric matrix
    M = np.random.randint(-100, 100, size=(n, n))
    M = M + M.T
    mx = np.sum(np.abs(M), axis=1)
    M = M + np.diag(mx)  
    return M

def produceEigen(M):
    # create eigenvalues and eigenvectors of M
    eigenvalues, eigenvectors = np.linalg.eig(M)
    return eigenvalues, eigenvectors

def constructMatrix(eigenvalues, eigenvectors):
    # construct matrix from eigenvalues and eigenvectors
    M = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), eigenvectors.T))
    return M

def checkMatrix(M, M_):
    # check if M is equal to M_
    return np.allclose(M, M_)

def main():
    n = int(input("Enter a number: "))
    if n < 1:
        print("n must be positive")
        return
    M = produceSymmMatrix(n)
    print("Produced Matrix: \n", M)
    eigenvalues, eigenvectors = produceEigen(M)
    print("Eigenvalues: \n", eigenvalues)
    print("Eigenvectors: \n", eigenvectors)
    M_ = constructMatrix(eigenvalues, eigenvectors)
    print("Is reconstruction okay: ", checkMatrix(M, M_))

if __name__ == "__main__":
    main()