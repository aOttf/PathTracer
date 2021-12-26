import Samplings as sl
import LinearAlg as la
import Renderings as rd
import numpy as np

import math

class DiffuseBRDF:
    """
    Diffuse material.
    Attributes:
        albedo: diffuse albedo.
    """

    def __init__(self, albedo):
        self.albedo = np.asarray(albedo)

    """
    The BRDF function.
    Parameters:
        hit: hitting point.
        wi: incident light direction in surface patch's local frame.
        wr: reflection direction in surface patch's local frame.
    """

    def eval(self, hit, wi, wr):
        frame = rd.Frame(np.array([0, 0, 1]))
        return self.albedo / math.pi * frame.cosTheta(wi)

    """
    BRDF importance sampling.
    Parameter:
        sampler: a Sampler object.
        wr: reflection direction in surface patch's local frame.
    Return (val, pdf, wi):
        val: BRDF value.
        pdf: pdf value of the sample.
        wi: the sampled incident light direction.
    """

    def sample(self, sampler, wr):
        val = np.zeros(3)
        wi = sl.cosineSampleHemisphere(sampler.next2D())
        pdf = sl.cosineHemispherePdf(wi)
        if pdf > 0:
            val = self.eval(np.zeros(3), wi, wr)
        return (val, pdf, wi)

    """
        pdf value at wi direction.
    """

    def pdf(self, wi, wr):
        return sl.cosineHemispherePdf(wi)


class PhongBRDF:
    """
    Phong material.
    Attributes:
        kd: diffuse reflection constant.
        ks: specular reflection constant.
        n: shininess.
    """

    def __init__(self, kd, ks, n):
        self.kd = np.array(kd)
        self.ks = np.array(ks)
        self.n = n
        dAvg = (kd[0] + kd[1] + kd[2]) / 3
        sAvg = (ks[0] + ks[1] + ks[2]) / 3
        self.specularSamplingWeight = sAvg / (dAvg + sAvg)

    """
    The BRDF function.
    Parameters:
        hit: hitting point.
        wi: incident light direction in surface patch's local frame.
        wr: reflection direction in surface patch's local frame.
    """

    def eval(self, hit, wi, wr):
        frame = rd.Frame(np.array([0, 0, 1]))
        return self.kd / math.pi + self.ks * (self.n + 2) / math.pi * math.pow(max(np.dot(wi, la.reflect(wr)), 0),
                                                                               self.n) * frame.cosTheta(wi)

    """
    BRDF importance sampling.
    Parameter:
        sampler: a Sampler object.
        wr: reflection direction in surface patch's local frame.
    Return (val, pdf, wi):
        val: BRDF value.
        pdf: pdf value of the sample.
        wi: the sampled incident light direction.
    """

    def sample(self, sampler, wr):
        val = np.zeros(3)
        prob_spec = self.specularSamplingWeight
        prob = sampler.next()
        wi_phong = sl.uniformSamplePhongLobe(sampler.next2D(), self.n)
        if prob < prob_spec:
            lobe_frame = rd.Frame(la.reflect(wr))
            wiWorld = lobe_frame.toWorld(wi_phong)
            frame = rd.Frame(np.array([0, 0, 1]))
            wi = frame.toLocal(wiWorld)
        else:
            wi = sl.cosineSampleHemisphere(sampler.next2D())

        pdf = self.pdf(wi_phong, wr)
        if pdf > 0:
            val = self.eval(np.zeros(3), wi, wr)

        return (val, pdf, wi)

    # note that v is wi in Phong lobe's local frame
    def pdf(self, wi, wr):
        return self.specularSamplingWeight * sl.uniformPhongLobePdf(wi, self.n) + (
                1 - self.specularSamplingWeight) * sl.cosineHemispherePdf(wi)