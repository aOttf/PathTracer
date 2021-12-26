import math
import time

import numpy as np
import matplotlib.pyplot as plt

import LinearAlg as la
import Renderings as rd
import Lights
import Materials
import Shapes

sampler = rd.Sampler(123456789)

def loadVeachScene(width, height, integrator, spp, maxDepth):
    objects = [
        # Light sources
        Shapes.Sphere([-3.75, 0, 0], 0.016666, None, [900, 200, 200]),
        Shapes.Sphere([-1.25, 0, 0], 0.05, None, [0, 100, 50]),
        Shapes.Sphere([1.25, 0, 0], 0.15, None, [0, 5, 10]),
        Shapes.Sphere([3.75, 0, 0], 0.5, None, [1, 1, 1]),
        Shapes.Sphere([10, 10, 4], 0.5, None, [800, 800, 800]),  # a bright light to provide the "ambient" light
        # Smooth plate
        Shapes.Triangle([4.000000, -2.706510, 0.256090], [4.000000, -2.083750, -0.526323], [-4.000000, -2.083750, -0.526323],
                 Materials.PhongBRDF([0, 0, 0], [1, 1, 1], 10000)),
        Shapes.Triangle([-4.000000, -2.083750, -0.526323], [-4.000000, -2.706510, 0.256090], [4.000000, -2.706510, 0.256090],
                 Materials.PhongBRDF([0, 0, 0], [1, 1, 1], 10000)),
        # Glossy plate
        Shapes.Triangle([4.000000, -3.288250, 1.369720], [4.000000, -2.838560, 0.476536], [-4.000000, -2.838560, 0.476536],
                 Materials.PhongBRDF([0, 0, 0], [1, 1, 1], 1000)),
        Shapes.Triangle([-4.000000, -2.838560, 0.476536], [-4.000000, -3.288250, 1.369720], [4.000000, -3.288250, 1.369720],
                 Materials.PhongBRDF([0, 0, 0], [1, 1, 1], 1000)),
        # Rough plate
        Shapes.Triangle([4.000000, -3.730960, 2.700457], [4.000000, -3.433780, 1.745637], [-4.000000, -3.433780, 1.745637],
                 Materials.PhongBRDF([0, 0, 0], [1, 1, 1], 100)),
        Shapes.Triangle([-4.000000, -3.433780, 1.745637], [-4.000000, -3.730960, 2.700457], [4.000000, -3.730960, 2.700457],
                 Materials.PhongBRDF([0, 0, 0], [1, 1, 1], 100)),
        # Super rough plate
        Shapes.Triangle([4.000000, -3.996153, 4.066700], [4.000000, -3.820690, 3.082207], [-4.000000, -3.820690, 3.082207],
                 Materials.PhongBRDF([0, 0, 0], [1, 1, 1], 50)),
        Shapes.Triangle([-4.000000, -3.820690, 3.082207], [-4.000000, -3.996153, 4.066700], [4.000000, -3.996153, 4.066700],
                 Materials.PhongBRDF([0, 0, 0], [1, 1, 1], 50)),
        # Floor
        Shapes.Triangle([-10.000000, -4.146147, -10.000003], [-10.000000, -4.146154, 9.999997],
                 [10.000000, -4.146154, 9.999997], Materials.DiffuseBRDF([0.5, 0.5, 0.5])),
        Shapes.Triangle([10.000000, -4.146154, 9.999997], [10.000000, -4.146147, -10.000003],
                 [-10.000000, -4.146147, -10.000003], Materials.DiffuseBRDF([0.5, 0.5, 0.5])),
        # Wall
        Shapes.Triangle([-10.000000, -10.000000, -2.000006], [10.000000, -10.000000, -2.000006],
                 [10.000000, 10.000000, -1.999994], Materials.DiffuseBRDF([0.5, 0.5, 0.5])),
        Shapes.Triangle([10.000000, 10.000000, -1.999994], [-10.000000, 10.000000, -1.999994],
                 [-10.000000, -10.000000, -2.000006], Materials.DiffuseBRDF([0.5, 0.5, 0.5]))
    ]
    lights = [Lights.SphereLight(objects[0]), Lights.SphereLight(objects[1]), Lights.SphereLight(objects[2]), Lights.SphereLight(objects[3]),
              Lights.SphereLight(objects[4])]
    camera = rd.Camera(np.array([0.0, 2.0, 15.0]), np.array([0.0, -2.0, 2.5]), np.array([0, 1, 0]))
    config = {'width': width, 'height': height, 'fov': 38, 'integrator': integrator, 'spp': spp, 'maxDepth': maxDepth}
    return rd.Scene(objects, lights, camera, config)

def loadCornellBox(width, height, integrator, spp, maxDepth):
    objects = [
        # Light
        Shapes.Triangle([343.0, 548.7999, 227.0], [343.0, 548.7999, 332.0], [213.0, 548.7999, 332.0], None, [60, 60, 60]),
        Shapes.Triangle([343.0, 548.7999, 227.0], [213.0, 548.7999, 332.0], [213.0, 548.7999, 227.0], None, [60, 60, 60]),
        # Floor
        Shapes.Triangle([552.8, 0, 0], [0, 0, 0], [0, 0, 559.2], Materials.DiffuseBRDF([0.885809, 0.698859, 0.666422])),
        Shapes.Triangle([552.8, 0, 0], [0, 0, 559.2], [552.8, 0, 559.2], Materials.DiffuseBRDF([0.885809, 0.698859, 0.666422])),
        # Ceiling
        Shapes.Triangle([556.0, 548.8, 0.0], [556.0, 548.8, 559.2], [0.0, 548.8, 559.2],
                 Materials.DiffuseBRDF([0.885809, 0.698859, 0.666422])),
        Shapes.Triangle([556.0, 548.8, 0.0], [0.0, 548.8, 559.2], [0.0, 548.8, 0.0],
                 Materials.DiffuseBRDF([0.885809, 0.698859, 0.666422])),
        # Back Wall
        Shapes.Triangle([549.6, 0.0, 559.2], [0.0, 0.0, 559.2], [0.0, 548.8, 559.2],
                 Materials.DiffuseBRDF([0.885809, 0.698859, 0.666422])),
        Shapes.Triangle([549.6, 0.0, 559.2], [0.0, 548.8, 559.2], [556.0, 548.8, 559.2],
                 Materials.DiffuseBRDF([0.885809, 0.698859, 0.666422])),
        # Right Wall
        Shapes.Triangle([0.0, 0.0, 559.2], [0.0, 0.0, 0.0], [0.0, 548.8, 0.0], Materials.DiffuseBRDF([0.1, 0.37798, 0.07])),
        Shapes.Triangle([0.0, 0.0, 559.2], [0.0, 548.8, 0.0], [0.0, 548.8, 559.2], Materials.DiffuseBRDF([0.1, 0.37798, 0.07])),
        # Left Wall
        Shapes.Triangle([552.8, 0.0, 0.0], [549.6, 0.0, 559.2], [556.0, 548.8, 559.2], Materials.DiffuseBRDF([0.57, 0.04, 0.04])),
        Shapes.Triangle([552.8, 0.0, 0.0], [556.0, 548.8, 559.2], [556.0, 548.8, 0.0], Materials.DiffuseBRDF([0.57, 0.04, 0.04])),
        # Short Block
        Shapes.Triangle([130.0, 165.0, 65.0], [82.0, 165.0, 225.0], [240.0, 165.0, 272.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([130.0, 165.0, 65.0], [240.0, 165.0, 272.0], [290.0, 165.0, 114.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([290.0, 0.0, 114.0], [290.0, 165.0, 114.0], [240.0, 165.0, 272.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([290.0, 0.0, 114.0], [240.0, 165.0, 272.0], [240.0, 0.0, 272.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([130.0, 0.0, 65.0], [130.0, 165.0, 65.0], [290.0, 165.0, 114.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([130.0, 0.0, 65.0], [290.0, 165.0, 114.0], [290.0, 0.0, 114.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([82.0, 0.0, 225.0], [82.0, 165.0, 225.0], [130.0, 165.0, 65.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([82.0, 0.0, 225.0], [130.0, 165.0, 65.0], [130.0, 0.0, 65.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([240.0, 0.0, 272.0], [240.0, 165.0, 272.0], [82.0, 165.0, 225.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([240.0, 0.0, 272.0], [82.0, 165.0, 225.0], [82.0, 0.0, 225.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        # Tall Block
        Shapes.Triangle([423.0, 330.0, 247.0], [265.0, 330.0, 296.0], [314.0, 330.0, 456.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([423.0, 330.0, 247.0], [314.0, 330.0, 456.0], [472.0, 330.0, 406.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([423.0, 0.0, 247.0], [423.0, 330.0, 247.0], [472.0, 330.0, 406.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([423.0, 0.0, 247.0], [472.0, 330.0, 406.0], [472.0, 0.0, 406.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([472.0, 0.0, 406.0], [472.0, 330.0, 406.0], [314.0, 330.0, 456.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([472.0, 0.0, 406.0], [314.0, 330.0, 456.0], [314.0, 0.0, 456.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([314.0, 0.0, 456.0], [314.0, 330.0, 456.0], [265.0, 330.0, 296.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([314.0, 0.0, 456.0], [265.0, 330.0, 296.0], [265.0, 0.0, 296.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([265.0, 0.0, 296.0], [265.0, 330.0, 296.0], [423.0, 330.0, 247.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([265.0, 0.0, 296.0], [423.0, 330.0, 247.0], [423.0, 0.0, 247.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85]))
    ]
    lights = [Lights.QuadLight(objects[0].v0, objects[0].v1, objects[0].v2, objects[1].v2, objects[0].radiance)]
    camera = rd.Camera(np.array([278, 273, -800]), np.array([278, 273, 0]), np.array([0, 1, 0]))
    config = {'width': width, 'height': height, 'fov': 38, 'integrator': integrator, 'spp': spp, 'maxDepth': maxDepth}
    return rd.Scene(objects, lights, camera, config)

def loadCornellBoxShapes(width, height, integrator, spp, maxDepth):
    objects = [
        # Light
        Shapes.Triangle([343.0, 548.7999, 227.0], [343.0, 548.7999, 332.0], [213.0, 548.7999, 332.0], None, [60, 60, 60]),
        Shapes.Triangle([343.0, 548.7999, 227.0], [213.0, 548.7999, 332.0], [213.0, 548.7999, 227.0], None, [60, 60, 60]),
        # Floor
        Shapes.Triangle([552.8, 0, 0], [0, 0, 0], [0, 0, 559.2], Materials.DiffuseBRDF([0.885809, 0.698859, 0.666422])),
        Shapes.Triangle([552.8, 0, 0], [0, 0, 559.2], [552.8, 0, 559.2], Materials.DiffuseBRDF([0.885809, 0.698859, 0.666422])),
        # Ceiling
        Shapes.Triangle([556.0, 548.8, 0.0], [556.0, 548.8, 559.2], [0.0, 548.8, 559.2],
                 Materials.DiffuseBRDF([0.885809, 0.698859, 0.666422])),
        Shapes.Triangle([556.0, 548.8, 0.0], [0.0, 548.8, 559.2], [0.0, 548.8, 0.0],
                 Materials.DiffuseBRDF([0.885809, 0.698859, 0.666422])),
        # Back Wall
        Shapes.Triangle([549.6, 0.0, 559.2], [0.0, 0.0, 559.2], [0.0, 548.8, 559.2],
                 Materials.DiffuseBRDF([0.885809, 0.698859, 0.666422])),
        Shapes.Triangle([549.6, 0.0, 559.2], [0.0, 548.8, 559.2], [556.0, 548.8, 559.2],
                 Materials.DiffuseBRDF([0.885809, 0.698859, 0.666422])),
        # Right Wall
        Shapes.Triangle([0.0, 0.0, 559.2], [0.0, 0.0, 0.0], [0.0, 548.8, 0.0], Materials.DiffuseBRDF([0.1, 0.37798, 0.07])),
        Shapes.Triangle([0.0, 0.0, 559.2], [0.0, 548.8, 0.0], [0.0, 548.8, 559.2], Materials.DiffuseBRDF([0.1, 0.37798, 0.07])),
        # Left Wall
        Shapes.Triangle([552.8, 0.0, 0.0], [549.6, 0.0, 559.2], [556.0, 548.8, 559.2], Materials.DiffuseBRDF([0.57, 0.04, 0.04])),
        Shapes.Triangle([552.8, 0.0, 0.0], [556.0, 548.8, 559.2], [556.0, 548.8, 0.0], Materials.DiffuseBRDF([0.57, 0.04, 0.04])),
        # Right Shapes.Sphere
        Shapes.Sphere([160, 82.5, 193], 82.5, Materials.PhongBRDF([0.425, 0.41, 0.38], [0.3, 0.3, 0.3], 128)),
        # Left Block
        Shapes.Triangle([423.0, 330.0, 247.0], [265.0, 330.0, 296.0], [314.0, 330.0, 456.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([423.0, 330.0, 247.0], [314.0, 330.0, 456.0], [472.0, 330.0, 406.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([423.0, 0.0, 247.0], [423.0, 330.0, 247.0], [472.0, 330.0, 406.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([423.0, 0.0, 247.0], [472.0, 330.0, 406.0], [472.0, 0.0, 406.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([472.0, 0.0, 406.0], [472.0, 330.0, 406.0], [314.0, 330.0, 456.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([472.0, 0.0, 406.0], [314.0, 330.0, 456.0], [314.0, 0.0, 456.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([314.0, 0.0, 456.0], [314.0, 330.0, 456.0], [265.0, 330.0, 296.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([314.0, 0.0, 456.0], [265.0, 330.0, 296.0], [265.0, 0.0, 296.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([265.0, 0.0, 296.0], [265.0, 330.0, 296.0], [423.0, 330.0, 247.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85])),
        Shapes.Triangle([265.0, 0.0, 296.0], [423.0, 330.0, 247.0], [423.0, 0.0, 247.0], Materials.DiffuseBRDF([0.85, 0.85, 0.85]))
    ]
    lights = [Lights.QuadLight(objects[0].v0, objects[0].v1, objects[0].v2, objects[1].v2, objects[0].radiance)]
    camera = rd.Camera(np.array([278, 273, -800]), np.array([278, 273, 0]), np.array([0, 1, 0]))
    config = {'width': width, 'height': height, 'fov': 38, 'integrator': integrator, 'spp': spp, 'maxDepth': maxDepth}
    return rd.Scene(objects, lights, camera, config)

def intersect(scene, ray):
    """
    Method to intersect all objects in the scene
    Parameters:
        scene: scene to render.
        ray: ray to trace.
    Returns:
      (has_hit, hit_object_id, distance_to_object)
      has_hit is 0 if there is not hit, else it is 1
      id is the idx of the scene object hit
      t is the ray parameter (recall a ray is o + td) at which
        a hit is found
    """
    n = len(scene.geometry)
    inf = t = 1e20
    d = 0
    id = -1
    for i in range(n):
        d = scene.geometry[i].intersect(ray)
        if d > 0 and d < t:
            t = d
            id = i
    return t < inf, id, t
def trace(scene, ray, sampler):
    """
    Trace a ray in the scene, and shade the hit point
    Parameters:
      scene: scene to render
      ray: ray to trace
    Returns:
      light radiance at the hit point
    """

    intersection = intersect(scene, ray)
    hasHit = intersection[0]
    if not hasHit:
        return np.zeros(3)
    hitObjectId = intersection[1]  # object id
    t = intersection[2]  # distance

    if scene.config['integrator'] == 'path_implicit':
        return traceImplicit(scene, sampler, ray.o + t * ray.d, -ray.d, scene.geometry[hitObjectId])
    elif scene.config['integrator'] == 'path_explicit':
        return traceExplicit(scene, sampler, ray.o + t * ray.d, -ray.d, scene.geometry[hitObjectId])
    else:
        raise ValueError(f'Unkonwn integrator \'{scene.config["integrator"]}\'.')

def render(scene, sampler):
    """
    Ray tracer main rendering function
    Parameters:
      scene: scene to render
    Returns:
      rendered image
    """

    spp = scene.config["spp"]
    cam = scene.cam  # cam pos and dir
    resx = scene.config["width"]
    resy = scene.config["height"]
    fov = scene.config["fov"]

    aspect_ratio = resx / resy
    scaling = np.tan(math.pi * fov / 360.0)
    invView = np.linalg.inv(la.lookAt(cam.eye, cam.at, cam.up))

    imgBuffer = np.zeros((resx, resy, 3))
    for x in range(resx):
        for y in range(resy):
            color = np.zeros(3)
            for s in range(spp):  # Additional loop to perform supersampling AA
                n = 1
                (a, b) = diskSampling(sampler)
                i, j = x + a, y + b
                t = n * scaling
                r = t * aspect_ratio
                u, v = -r + ((i + 0.5) * 2 * r / resx), t - ((j + 0.5) * 2 * t / resy)
                dir = la.normalize(u * invView[:-1, 0] + v * invView[:-1, 1] - n * invView[:-1, 2])
                ray = rd.Ray(cam.eye, dir)
                color += trace(scene, ray, sampler)

            color /= spp
            imgBuffer[y][x] = color

    return imgBuffer

# Rejection Sampling a Disk
def diskSampling(sampler):
    while True:
        a = (1 - 2 * sampler.next()) / 2
        b = (1 - 2 * sampler.next()) / 2
        if(a**2 + b**2 < .25):
            break
    return (a, b)

def traceImplicit(scene, sampler, hitPoint, hitDir, hitObject):
    if np.any(hitObject.radiance):
        return hitObject.radiance
    return traceImplicit_Rec(scene, sampler, hitPoint, hitDir, hitObject, scene.config['maxDepth'])

"""
    Main Recursion of Implicit Tracing
    Parameters:
        scene: a Scene object.
        sampler: a Sampler object for sampling.
        hitPoint: the hit point x in the rendering equation.
        hitDir: the outgoing direction w in the rendering equation.
        hitObject: the hit object (e.g., a Shapes.Triangle instance).
        depthRem: the max recursion times remaining
    Return:
        Lr: the radiance L(x, w).
"""
def traceImplicit_Rec(scene, sampler, hitPoint, hitDir, hitObject, depthRem):
    # whether reaches max depth
    if depthRem < 0:
        return np.zeros(3)

    # init Lr
    Lr = np.zeros(3)

    # whether hits light source
    if np.any(hitObject.radiance):
        return hitObject.radiance

    #otherwise, recursion
    wrWorld = hitDir
    wrLocal = hitObject.frame.toLocal(wrWorld)

    # val = fr * cosine
    (val, pdf, wiLocal) = hitObject.mat.sample(sampler, wrLocal)
    wiWorld = hitObject.frame.toWorld(wiLocal)

    #generate new ray for tracing
    shadowRay = rd.Ray(hitPoint + wiWorld * .001, wiWorld)
    shadowInfo = intersect(scene, shadowRay)

    #handle shadowing
    if shadowInfo[0]:
       # print("hit")
        # successfully hit a non-light source object
        shadowhitPoint = shadowRay.o + shadowRay.d * shadowInfo[2]
        shadowhitDir = -shadowRay.d
        shadowhitObject = scene.geometry[shadowInfo[1]]
        depthRem -= 1

        #recursively call the function
        Lr = traceImplicit_Rec(scene, sampler, shadowhitPoint, shadowhitDir, shadowhitObject, depthRem) * val / pdf
    return Lr

"""
    Compute and return the radiance contributed by the direct illumination L_dir(x, w) using light importance sampling in the area domain.
    Parameters:
        scene: a Scene object.
        sampler: a Sampler object for sampling.
        hitPoint: the hit point x in the rendering equation.
        hitDir: the outgoing direction w in the rendering equation.
        hitObject: the hit object (e.g., a Shapes.Triangle instance).
    Return:
        Lr: the radiance contributed by the direct illumination L_dir(x, w).
"""
def sampleLight(scene, sampler, woWorld, hitObject, hitPoint):
    # TODO: YOU CODE HERE.
    Lr_light = np.zeros(3)

    if np.any(hitObject.radiance):
        return hitObject.radiance

    # Initialize variable to store outgoing radiance
    (light, pdf_select) = Lights.selectLight(scene, sampler)
    (pos, normal, pdf) = light.sampleByArea(sampler)
    pdf *= pdf_select

    # Distance
    r = la.squaredDistance(pos, hitPoint)
    wiWorld = la.normalize(pos - hitPoint)
    wiLocal = hitObject.toLocal(wiWorld, hitPoint)

    shadowRay = rd.Ray(hitPoint + wiWorld * 0.001, wiWorld)
    shadowInfo = intersect(scene, shadowRay)
    # Handle shadows
    if shadowInfo[0]:
        cosTheta0 = np.dot(normal, -wiWorld)
        Li = scene.geometry[shadowInfo[1]].radiance
        wrLocal = hitObject.toLocal(woWorld, hitPoint)
        G = (abs(cosTheta0)) / r
        Lr_light = Li * hitObject.mat.eval(hitPoint, wiLocal, wrLocal) * G/pdf


    return Lr_light

def sampleLight_SolidAngle(scene, sampler, woWorld, hitObject, hitPoint):
    Lr_light = np.zeros(3)

    if np.any(hitObject.radiance):
        return hitObject.radiance

    # Initialize variable to store outgoing radiance
    (light, pdf_select) = Lights.selectLight(scene, sampler)
    wiWorld = light.sampleBySolidAngle(sampler, hitPoint)
    pdf = light.pdfSolidAngle(sampler, hitPoint)
    pdf = pdf * pdf_select
    wiLocal = hitObject.frame.toLocal(wiWorld)
    shadowRay = rd.Ray(hitPoint + wiWorld * 0.001, wiWorld)
    shadowInfo = intersect(scene, shadowRay)
    # Handle shadows
    if shadowInfo[0]:
        Li = scene.geometry[shadowInfo[1]].radiance
        wrWorld = la.normalize(woWorld)
        wrLocal = hitObject.frame.toLocal(woWorld)
        Lr_light = Li * hitObject.mat.eval(hitPoint, wiLocal, wrLocal) / pdf

    return Lr_light

"""
    Compute and return the radiance L(x, w) using explicit path tracing.
    The number of bounces of the path should not exceed scene.config['maxDepth'].
    Parameters:
        scene: a Scene object.
        sampler: a Sampler object for sampling.
        hitPoint: the hit point x in the rendering equation.
        hitDir: the outgoing direction w in the rendering equation.
        hitObject: the hit object (e.g., a Shapes.Triangle instance).
    Return:
        Lr: the radiance L(x, w).
"""
def traceExplicit(scene, sampler, hitPoint, hitDir, hitObject):

    maxDepth = scene.config['maxDepth']

    # directly hit light source
    if np.any(hitObject.radiance):
        return hitObject.radiance

    Lr = traceExplicit_Rec(scene, sampler, hitPoint, hitDir, hitObject, maxDepth)
    return Lr

def traceExplicit_Rec(scene, sampler, hitPoint, hitDir, hitObject, depthRem):
    # reaches max depth
    if depthRem < 0:
        return np.zeros(3)

    # hits light source, avoid double counting
    if np.any(hitObject.radiance):
        return np.zeros(3)


    Lind = np.zeros(3)
    #Ldir = np.zeros(3)
    Ldir = sampleLight(scene, sampler, hitDir, hitObject, hitPoint)
    #print(Ldir)
    wrWorld = hitDir
    wrLocal = hitObject.toLocal(wrWorld, hitPoint)

    # val = fr * cosine
    (val, pdf, wiLocal) = hitObject.mat.sample(sampler, wrLocal)
    wiWorld = hitObject.toWorld(wiLocal, hitPoint)

    # generate new ray for tracing
    shadowRay = rd.Ray(hitPoint + wiWorld * .001, wiWorld)
    shadowInfo = intersect(scene, shadowRay)

    # handle shadowing
    if shadowInfo[0]:
        # successfully hit a non-light source object
        hitPoint = shadowRay.o + shadowRay.d * shadowInfo[2]
        hitDir = -shadowRay.d
        hitObject = scene.geometry[shadowInfo[1]]
        depthRem -= 1
        Lind = traceExplicit_Rec(scene, sampler, hitPoint, hitDir, hitObject, depthRem) * val / pdf

    return Ldir + Lind

def traceExplicit_Roulette(scene, sampler, hitPoint, hitDir, hitObject):
    # directly hit light source
    if np.any(hitObject.radiance):
        return hitObject.radiance
    q = .1
    Lr = traceExplicit_Roulette_Rec(scene, sampler, hitPoint, hitDir, hitObject, q)
    return Lr

def traceExplicit_Roulette_Rec(scene, sampler, hitPoint, hitDir, hitObject, q):
    # hits light source, avoid double counting
    if np.any(hitObject.radiance):
        return np.zeros(3)

    Lind = np.zeros(3)
    # Ldir = np.zeros(3)
    Ldir = sampleLight(scene, sampler, hitDir, hitObject, hitPoint)

    # decides whether terminates the current path
    if sampler.next() > q:
        wrWorld = hitDir
        wrLocal = hitObject.frame.toLocal(wrWorld)

        # val = fr * cosine
        (val, pdf, wiLocal) = hitObject.mat.sample(sampler, wrLocal)
        wiWorld = hitObject.frame.toWorld(wiLocal)

        # generate new ray for tracing
        shadowRay = rd.Ray(hitPoint + wiWorld * .001, wiWorld)
        shadowInfo = intersect(scene, shadowRay)

        # handle shadowing
        if shadowInfo[0]:
            # successfully hit a non-light source object
            hitPoint = shadowRay.o + shadowRay.d * shadowInfo[2]
            hitDir = -shadowRay.d
            hitObject = scene.geometry[shadowInfo[1]]
            Lind = traceExplicit_Roulette_Rec(scene, sampler, hitPoint, hitDir, hitObject, q) * val / pdf

    return Ldir + (Lind /(1-q))

if __name__ == "__main__":

    #Implicit Path Tracing
    imgWidth = 128
    imgHeight = 128
    spp = 256
    maxDepth = 2

    #Cornell Box
    scene = loadCornellBox(imgWidth, imgHeight, 'path_implicit', spp, maxDepth)
    print(f"Start rendering in resolution: {imgWidth}x{imgHeight}, fov: {scene.config['fov']}, \
        integrator: {scene.config['integrator']}, \
        spp: {scene.config['spp']}")
    startTime = time.monotonic()
    img = render(scene, sampler)
    duration = time.monotonic() - startTime
    print("Finished rendering, time taken:", duration, "seconds")

    renderResult = np.power(np.clip(img, 0.0, 1.0), 1 / 2.2)

    plt.imshow(renderResult)
    plt.show()
    plt.imsave("Implicit-Box")


    # Explicit Path Tracing
    imgWidth = 128
    imgHeight = 128
    spp = 256
    maxDepth = 2

    #Cornell Box
    scene = loadCornellBox(imgWidth, imgHeight, 'path_explicit', spp, maxDepth)
    print(f"Start rendering in resolution: {imgWidth}x{imgHeight}, fov: {scene.config['fov']}, \
        integrator: {scene.config['integrator']}, \
        spp: {scene.config['spp']}")
    startTime = time.monotonic()
    img = render(scene, sampler)
    duration = time.monotonic() - startTime
    print("Finished rendering, time taken:", duration, "seconds")

    renderResult = np.power(np.clip(img, 0.0, 1.0), 1 / 2.2)

    plt.imshow(renderResult)
    plt.show("Explicit-Box")


    imgWidth = 128
    imgHeight = 128
    spp = 64
    maxDepth = 4
    #Cornell Box Sphere
    scene = loadCornellBoxShapes.Sphere(imgWidth, imgHeight, 'path_explicit', spp, maxDepth)
    print(f"Start rendering in resolution: {imgWidth}x{imgHeight}, fov: {scene.config['fov']}, \
           integrator: {scene.config['integrator']}, \
           spp: {scene.config['spp']}")
    startTime = time.monotonic()
    img = render(scene, sampler)
    duration = time.monotonic() - startTime
    print("Finished rendering, time taken:", duration, "seconds")

    renderResult = np.power(np.clip(img, 0.0, 1.0), 1 / 2.2)

    plt.imshow(renderResult)
    plt.show()
    plt.imsave("Explicit-Shapes.Sphere")


"""
    #Veach
    imgWidth = imgHeight = 128
    scene = loadVeachScene(imgWidth, imgHeight, 'path_explicit', 64, 4)
    print(f"Start rendering in resolution: {imgWidth}x{imgHeight}, fov: {scene.config['fov']}, \
        integrator: {scene.config['integrator']}, \
        spp: {scene.config['spp']}")
    startTime = time.monotonic()
    img = render(scene, sampler)
    duration = time.monotonic() - startTime
    print("Finished rendering, time taken:", duration, "seconds")

    renderResult = np.power(np.clip(img, 0.0, 1.0), 1 / 2.2)

    plt.imshow(renderResult)
    plt.show()
    plt.imsave("Explicit-Veach")
"""





