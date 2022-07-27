import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import ipdb
import timeit


device = "cuda"

def best_fit_transform_torch(A,B):
    '''
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
          A: Nxm numpy array of corresponding points
          B: Nxm numpy array of corresponding points
        Returns:
          T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
          R: mxm rotation matrix
          t: mx1 translation vector
        '''

    assert A.shape == B.shape
    torch.no_grad()
    # get number of dimensions
    m = A.shape[1]
    is_torch_A = isinstance(A, torch.Tensor)
    if not is_torch_A:
        A = torch.from_numpy(A)
    is_torch_B = isinstance(B, torch.Tensor)
    if not is_torch_B:
        B = torch.from_numpy(B)
    A = A.to(device)
    B = B.to(device)
    # translate points to their centroids
    centroid_A = torch.mean(A, dim=0).unsqueeze(0).to(device)
    centroid_B = torch.mean(B, dim=0).unsqueeze(0).to(device)

    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = torch.mm(AA.T, BB)

    # U, S, Vt = torch.linalg.svd(H)
    U, S, Vt = torch.svd(H)
    R = torch.mm(Vt.T, U.T)

    # special reflection case
    if torch.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = torch.mm(Vt.T, U.T)

    # translation
    t = centroid_B.T - torch.mm(R, centroid_A.T)
    # homogeneous transformation
    T = torch.eye(m + 1,dtype=torch.double).to(device)
    T[:m, :m] = R
    T[:m, m] = t.squeeze()

    return T, R, t
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def nearest_neighbor_torch(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    src = torch.clone(src).cpu()
    dst = torch.clone(dst).cpu()
    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return torch.Tensor(distances.ravel()), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i

def icp_torch(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
        The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
        Input:
            A: Nxm numpy array of source mD points
            B: Nxm numpy array of destination mD point
            init_pose: (m+1)x(m+1) homogeneous transformation
            max_iterations: exit algorithm after max_iterations
            tolerance: convergence criteria
        Output:
            T: final homogeneous transformation that maps A on to B
            distances: Euclidean distances (errors) of the nearest neighbor
            i: number of iterations to converge
        '''

    assert A.shape == B.shape
    torch.no_grad()
    # get number of dimensions
    m = A.shape[1]

    is_torch_A = isinstance(A, torch.Tensor)
    if not is_torch_A:
        A = torch.from_numpy(A)
    is_torch_B = isinstance(B, torch.Tensor)
    if not is_torch_B:
        B = torch.from_numpy(B)

    # make points homogeneous, copy them to maintain the originals
    src = torch.ones((m + 1, A.shape[0]),dtype=torch.double)
    dst = torch.ones((m + 1, B.shape[0]),dtype=torch.double)
    src[:m, :] = torch.clone(A.T)
    dst[:m, :] = torch.clone(B.T)
    src = src.to(device)
    dst = dst.to(device)
    # apply the initial pose estimation
    if init_pose is not None:
        init_pose = torch.from_numpy(init_pose).to(device)
        src = torch.mm(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor_torch(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform_torch(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = torch.mm(T, src)

        # check error
        mean_error = torch.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform_torch(A, src[:m, :].T)

    return T.cpu(), distances, i


if __name__ =="__main__":
    A = np.arange(10*3).astype(float).reshape(-1,3)
    B = np.arange(10 * 3).astype(float).reshape(-1,3)
    start = timeit.default_timer()
    print(icp(A,B))
    se = timeit.default_timer()
    print(icp_torch(A,B))
    end = timeit.default_timer()
    print("use cpu [%f] use gpu [%f]" % ((se - start),(end-se)))
    # print(best_fit_transform_torch(A,B))