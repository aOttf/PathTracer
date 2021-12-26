import Samplings as sl
import LinearAlg as la
import Renderings as rd
import numpy as np

class QuadLight:
    def __init__(self, v0, v1, v2, v3, radiance):
        self.trig1 = [v0, v1, v2]
        self.trig2 = [v0, v2, v3]
        self.radiance = radiance
        self.area1 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        self.area = self.area1 + 0.5 * np.linalg.norm(np.cross(v2 - v0, v3 - v0))

    def sampleByArea(self, sampler):
        trig = self.trig1 if self.area1 / self.area >= sampler.next() else self.trig2
        uv = sl.uniformSampleTriangle(sampler.next2D())
        pos = trig[0] * (1 - uv[0] - uv[1]) + trig[1] * uv[0] + trig[2] * uv[1]
        normal = la.normalize(np.cross(trig[1] - trig[0], trig[2] - trig[0]))
        pdf = 1 / self.area
        return (pos, normal, pdf)


class SphereLight:
    def __init__(self, sphere):
        self.center = sphere.center
        self.radius = sphere.radius
        self.radiance = sphere.radiance

    def sampleBySolidAngle(self, sampler, hit):
        """
        Sample a direction towards the cone of directions subtended by this spherical light
        Parameter:
            sampler: a Sampler instance
            hit: the shading point in world frame
        Returns:
            wiWorld: the sampled light direction in world frame
        """
        # TODO: YOUR CODE HERE
        dist = la.distance(hit, self.center)
        wc = la.normalize(self.center - hit)
        frame = rd.Frame(-wc)

        # Compute thetaMax
        sinThetaMax = (self.radius ** 2) / (dist ** 2)
        cosThetaMax = (max(0., 1 - sinThetaMax)) ** 0.5
        res = sl.uniformSampleCone(sampler.next2D(), cosThetaMax)
        pos = frame.toLocal(res)
        pos *= self.radius
        pos += self.center
        return la.normalize(pos - hit)

    def pdfSolidAngle(self, sampler, hit):
        """
        Compute the PDF value for sampling a direction towards this spherical light
        Parameter:
            sampler: a Sampler instance
            hit: the shading point in world frame
        Returns:
            pdf: the computed PDF value
        """

        sinThetaMax = (self.radius ** 2) / la.squaredDistance(hit, self.center)
        cosThetaMax = (max(0., 1 - sinThetaMax)) ** 0.5

        return sl.uniformConePdf(cosThetaMax)

def selectLight(scene, sampler):
    idx = int(sampler.next() * len(scene.lights))
    idx = min(idx, len(scene.lights) - 1)
    pdf = 1 / len(scene.lights)
    return (scene.lights[idx], pdf)