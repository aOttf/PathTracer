import numpy as np
import LinearAlg as la, Renderings as rd


class Triangle:
    """
    A class to represent a triangle
    Attributes:
        v0, v1, v2: vertices of the triangle
        mat: material of the triangle
        radiance: (only applicable to quad lights) emission radiance of the triangle
    """

    def __init__(self, v0, v1, v2, mat, radiance=None):
        self.v0 = np.asarray(v0)
        self.v1 = np.asarray(v1)
        self.v2 = np.asarray(v2)
        self.mat = mat

        n = la.normalize(np.cross(self.v1 - self.v0, self.v2 - self.v0))
        self.frame = rd.Frame(n)

        if radiance is None:
            self.radiance = np.array([0, 0, 0])
        else:
            self.radiance = np.asarray(radiance)

    def intersect(self, r):
        """
        Intersect this triangle with a ray
        Parameters:
            r: ray to intersect
        Returns:
            0 if no hit, otherwise the distance to the hit point
        """
        e1 = self.v1 - self.v0
        e2 = self.v2 - self.v0
        h = np.cross(r.d, e2)
        a = np.dot(e1, h)
        if a > -1e-6 and a < 1e-6:
            return 0
        f = 1.0 / a
        s = r.o - self.v0
        u = f * np.dot(s, h)
        if u < 0 or u > 1:
            return 0
        q = np.cross(s, e1)
        v = f * np.dot(r.d, q)
        if v < 0 or u + v > 1:
            return 0
        t = f * np.dot(e2, q)
        if t > 1e-6:
            return 0 if t > r.max else t  # if exceed max distance, return 0
        else:
            return 0

    def normal(self, hit):
        """
        Computes the normal vector of the triangle at a given point
        Parameters:
            hit: the point where the normal needs to be computed
        Returns:
            the normal vector at the 'hit' point
        """
        return la.normalize(np.cross(self.v0 - hit, self.v1 - hit))

    def toWorld(self, v, hit):
        return self.frame.toWorld(v)

    def toLocal(self, v, hit):
        return self.frame.toLocal(v)


class Sphere:
    def __init__(self, center, radius, material, radiance=None):
        self.center = center
        self.radius = radius
        self.mat = material
        if radiance is None:
            self.radiance = np.array([0, 0, 0])
        else:
            self.radiance = np.asarray(radiance)

    def intersect(self, ray):
        A = np.dot(ray.d, ray.d)
        B = 2 * np.dot(ray.o - self.center, ray.d)
        C = np.dot(ray.o - self.center, ray.o - self.center) - self.radius * self.radius
        discriminant = B * B - 4 * A * C
        if discriminant < 0:
            return 0
        else:
            t1 = (-B - np.sqrt(discriminant)) / (2 * A)
            if t1 > 0:
                return t1
            t2 = (-B + np.sqrt(discriminant)) / (2 * A)
            if t2 > 0:
                return t2
            else:
                return 0

    def normal(self, hit):
        return la.normalize(hit - self.center)

    def toWorld(self, v, hit):
        frame = rd.Frame(self.normal(hit))
        return frame.toWorld(v)
    def toLocal(self, v, hit):
        frame = rd.Frame(self.normal(hit))
        return frame.toLocal(v)