import numpy as np

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

# resource is: http://nghiaho.com/?page_id=671
# https://math.stackexchange.com/questions/4010538/solve-for-2d-translation-rotation-and-scale-given-two-touch-point-movements

def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


if __name__ == '__main__':
    # Test with random data

    # Random rotation and translation
    R = np.random.rand(3,3)
    t = np.random.rand(3,1)

    # make R a proper rotation matrix, force orthonormal
    U, S, Vt = np.linalg.svd(R)
    R = U@Vt

    # remove reflection
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = U@Vt
        
        # number of points
        n = 10

        A = np.random.rand(3, n)
        B = R@A + t

        # Recover R and t
        ret_R, ret_t = rigid_transform_3D(A, B)

        # Compare the recovered R and t with the original
        B2 = (ret_R@A) + ret_t

        # Find the root mean squared error
        err = B2 - B
        err = err * err
        err = np.sum(err)
        rmse = np.sqrt(err/n)

        print("Points A")
        print(A)
        print("")

        print("Points B")
        print(B)
        print("")

        print("Ground truth rotation")
        print(R)

        print("Recovered rotation")
        print(ret_R)
        print("")

        print("Ground truth translation")
        print(t)

        print("Recovered translation")
        print(ret_t)
        print("")

        print("RMSE:", rmse)

        if rmse < 1e-5:
            print("Everything looks good!")
        else:
            print("Hmm something doesn't look right ...")