import math
import numpy as np

def concentricSampleDisk(sample):
    sampleOffset = 2 * sample - np.ones(2)
    if not any(sample):
        return np.zeros(2)
    if np.abs(sampleOffset[0]) > np.abs(sampleOffset[1]):
        r = sampleOffset[0]
        theta = (math.pi / 4) * (sampleOffset[1] / sampleOffset[0])
    else:
        r = sampleOffset[1]
        theta = math.pi / 2 - (math.pi / 4) * (sampleOffset[0] / sampleOffset[1])
    return r * np.array([np.cos(theta), np.sin(theta)])


def cosineSampleHemisphere(sample):
    d = concentricSampleDisk(sample)
    z = np.sqrt(max(0, 1 - d[0] * d[0] - d[1] * d[1]))
    return np.array([d[0], d[1], z])


def cosineHemispherePdf(v):
    return v[2] / math.pi if v[2] > 0 else 0


def uniformSampleCone(sample, cosThetaMax):
    cosTheta = (1 - sample[0]) + sample[0] * cosThetaMax
    sinTheta = max(0, np.sqrt(1 - cosTheta * cosTheta))
    phi = 2 * math.pi * sample[1]
    return np.array([sinTheta * np.cos(phi), sinTheta * np.sin(phi), cosTheta])


def uniformConePdf(cosThetaMax):
    return 1 / (2 * math.pi * (1 - cosThetaMax))


def uniformSampleTriangle(sample):
    u = np.sqrt(1 - sample[0])
    return np.array([1 - u, u * sample[1]])


def uniformSamplePhongLobe(sample, n):
    z = math.pow(1 - sample[0], 1 / (n + 1))
    r = math.sqrt(1 - z * z)
    phi = 2 * math.pi * sample[1]
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    return np.array([x, y, z])


def uniformPhongLobePdf(v, n):
    return (n + 1) / (2 * math.pi) * math.pow(v[2], n)
