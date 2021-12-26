import numpy as np

def normalize(v, eps=1e-8):
    """
    Normalize a vector. Add a tiny eps to the denominator to prevent
    divide-by-zero errors.
    """
    dist = np.linalg.norm(v)
    return v / (dist + eps)


def distance(a, b):
    return np.linalg.norm(b - a)


def squaredDistance(a, b):
    v = b - a
    return np.dot(v.T, v)


def lookAt(eye, at, up):
    """
    Viewing transformation.
    Parameters:
        eye (np.array): eye postion
        at (np.array): the point the eye is looking at (usually, the center of
          an object of interest)
        up (np.array): up vector (vertically upward direction)
    """
    z = normalize(eye - at)
    x = normalize(np.cross(up, z))
    y = normalize(np.cross(z, x))
    A = np.column_stack((x, y, z, eye))
    A = np.row_stack((A, np.array([0, 0, 0, 1])))
    return np.linalg.inv(A)


def reflect(v):
    """
    Reflect a vector in local frame.
    """
    return np.array([-v[0], -v[1], v[2]])
