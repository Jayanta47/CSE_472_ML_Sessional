import numpy as np

def produceMatrix(n):
    # create random invertible integer matrix of size n, n
    M = np.random.randint(-100, 100, size=(n, n))
    mx = np.sum(np.abs(M), axis=1)
    M = M + np.diag(mx)    
    return M

def produceEigen(M):
    # create eigenvalues and eigenvectors of M
    eigenvalues, eigenvectors = np.linalg.eig(M)
    return eigenvalues, eigenvectors

def constructMatrix(eigenvalues, eigenvectors):
    # construct matrix from eigenvalues and eigenvectors
    M = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), np.linalg.inv(eigenvectors)))
    return M

def checkMatrix(M, M_):
    # check if M is equal to M_
    return np.allclose(M, M_)

def main():
    n = int(input("Enter a number: "))
    if n < 1:
        print("n must be positive")
        return
    M = produceMatrix(n)
    print("Produced Matrix: \n", M)
    eigenvalues, eigenvectors = produceEigen(M)
    print("Eigenvalues: \n", eigenvalues)
    print("Eigenvectors: \n", eigenvectors)
    M_ = constructMatrix(eigenvalues, eigenvectors)
    print("Is reconstruction okay: ", checkMatrix(M, M_))


if __name__ == "__main__":
    main()
