import numpy as np
import random

class Scene:
    """
    A class used to represent a rendering scene
    Attributes:
        geometry: objects in the scene.
        lights: light info
        camera: camera info
        config: store width, height and fov
    """

    def __init__(self, geometry, lights, camera, config):
        self.geometry = geometry
        self.lights = lights
        self.cam = camera
        # config is a dictionary with keys: width, height, fov, samplingStrategy, numSamples
        self.config = config


class Camera:
    """
    A class to represent a camera
    Attributes:
        eye: position of the camera
        at: position where the camera looks at
        up: up direction of the camera
    """

    def __init__(self, eye, at, up):
        self.eye = eye
        self.at = at
        self.up = up


class Ray:
    """
    A class to represent a ray
    Attributes:
        o: origin of the ray
        d: direction of the ray
        max: max distance at which the ray is capped (to avoid evaluating
          indefinitely far points along a ray, we cap it to a large distance)
    """

    def __init__(self, origin, dir, max=1e5):
        self.o = origin
        self.d = dir
        self.max = max


class Frame:
    """
    A class to represent a coordinate frame of the given normal
    Attributes:
        normal: normal vector in world space
    """

    def __init__(self, normal):
        self.n = normal
        if abs(normal[0]) > abs(normal[1]):
            invLen = 1 / np.sqrt(normal[0] * normal[0] + normal[2] * normal[2])
            self.t = np.array([normal[2] * invLen, 0, -normal[0] * invLen])
        else:
            invLen = 1 / np.sqrt(normal[1] * normal[1] + normal[2] * normal[2])
            self.t = np.array([0, normal[2] * invLen, -normal[1] * invLen])
        self.s = np.cross(self.t, self.n)

    def toLocal(self, v):
        """Convert an input vector v from the world fram to the local frame. """
        return np.array([np.dot(v, self.s), np.dot(v, self.t), np.dot(v, self.n)])

    def toWorld(self, v):
        """Convert an input vector v from the local fram to the world frame. """
        return self.s * v[0] + self.t * v[1] + self.n * v[2]

    def cosTheta(self, v):
        """ cosine of the angle between vector v and the normal vector of the frame. """
        return v[2]


class Sampler:
    """
    A base class that implements a sampler
    Attributes:
        seed: sampler seed
    """

    def __init__(self, seed):
        random.seed(seed)

    def next(self):
        """Draw the next sample (scalar). """
        return random.random()

    def next2D(self):
        """Draw the next 2D sample. """
        return np.array([random.random(), random.random()])