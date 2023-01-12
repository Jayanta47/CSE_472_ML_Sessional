import numpy as np 

def produceMatrix(n, m):
    M = np.random.randint(-100, 100, size=(n, m))  
    return M

def moorePenrose(M):
    # Moore-Penrose pseudoinverse of matrix M

    # singular value decompositon of M
    U, D, V = np.linalg.svd(M, full_matrices=False)
    # print("U: \n", U)
    # print("D: \n", D)
    # print("V: \n", V)
    D = np.diag(D)
    D_ = np.linalg.inv(D)
    # print(np.zeros((U.shape[0] - D_.shape[1], 1), dtype=int).T)
    # # make D_ same column as row size of U  
    # D_plus = np.concatenate((D_, np.zeros((U.shape[0] - D_.shape[1], D_.shape[0]), dtype=int).T), axis=1)
    A_ = np.dot(V.T, np.dot(D_, U.T))
    return A_


def checkMatrix(M, M_):
    # check if M is equal to M_
    return np.allclose(M, M_)

def main():
    n = int(input("Enter n: "))
    m = int(input("Enter m: "))
    if n < 0 or m < 0:
        print("n and m must be positive")
        return
    M = produceMatrix(n, m)
    print("Produced Matrix: \n", M)
    A_ = moorePenrose(M)
    print("Moore-Penrose Pseudoinverse: \n", A_)
    A_np = np.linalg.pinv(M)
    print("Are pseudo-inverses equal: ", checkMatrix(A_, A_np))

if __name__ == "__main__":
    main()