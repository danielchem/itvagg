from __future__ import division, print_function
from random import randint, choice
import math
import numpy

from visual import *
from scipy import ndimage

import argparse
parser = argparse.ArgumentParser(description='My app description')
parser.add_argument('-i', '--input', help='Path to input file')
parser.add_argument('-o', '--output', help='Path to output file')
args = parser.parse_args()
# from numpy import linalg 




__version__ = '2015.07.18'
__docformat__ = 'restructuredtext en'
__all__ = ()


def unit_vector(vec):
    return numpy.array(vec/numpy.linalg.norm(vec))

def calculate_COM(agg):
    """ Calclulate Center of Mass for the given aggregate
    """
    # print (agg.shape[1], agg.shape[0])
    return numpy.sum(agg, axis = 1) / agg.shape[1]

def calculate_LD(aggregate):
    longdist = 0.0
    for i in xrange(aggregate.shape[1]):
        for j in xrange(i+1, aggregate.shape[1]):
            temp = aggregate[:,i] - aggregate[:,j]
            longdist = max(longdist, numpy.sqrt(numpy.dot(temp.T, temp)))
    return longdist 


def identity_matrix():
    """Return 4x4 identity/unit matrix.

    >>> I = identity_matrix()
    >>> numpy.allclose(I, numpy.dot(I, I))
    True
    >>> numpy.sum(I), numpy.trace(I)
    (4.0, 4.0)
    >>> numpy.allclose(I, numpy.identity(4))
    True

    """
    return numpy.identity(4)


def translation_matrix(direction):
    """Return matrix to translate by direction vector.

    >>> v = numpy.random.random(3) - 0.5
    >>> numpy.allclose(v, translation_matrix(v)[:3, 3])
    True

    """
    M = numpy.identity(4)
    M[:3, 3] = direction[:3]
    return M

def transform_input_data(aggregate):
    return numpy.append(numpy.transpose(aggregate), [numpy.ones(aggregate.shape[0])], axis = 0)

def translate_aggregate(aggregate, translation_vec):
    return numpy.dot(translation_matrix(translation_vec), aggregate)

def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = numpy.diag([cosa, cosa, cosa])
    R += numpy.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += numpy.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = numpy.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
        M[:3, 3] = point - numpy.dot(R, point)
    return M

def rotate_aggregate(aggregate, angle):
    #angle = math.pi/2

    direction = numpy.array([1., 1., 1.])
    return numpy.dot(rotation_matrix(angle, direction, calculate_COM(aggregate)[:3]), aggregate)

def gen_rand_vecs(dims, number):
    vecs = numpy.random.normal(size=(number,dims))
    mags = numpy.linalg.norm(vecs, axis=-1)
    return numpy.array(vecs / mags[..., numpy.newaxis])[0,:]

def random_rotate_aggregate(aggregate):
    angle = numpy.radians(random.uniform(0, 360))
    direction = numpy.array([0., 0., 1.])
    return numpy.dot(rotation_matrix(angle, gen_rand_vecs(3,1), calculate_COM(aggregate)[:3]), aggregate)


def test_collision(agg1, agg2, partDiameter):
    check = 1
    for i in xrange(agg1.shape[1]):
        for j in xrange(agg2.shape[1]):
            dx = agg1[0,i] - agg2[0,j]
            dy = agg1[1,i] - agg2[1,j]
            dz = agg1[2,i] - agg2[2,j]
            dist = numpy.linalg.norm([dx, dy, dz])
            # j = 99
            if ( dist < 1.01*partDiameter ):
                check = 2
                # print (check, j)
                # print (poc)
                return check, j
                break
    return check, 99          
 
# random point in bounden domain

def random_point_generator(LD1, LD2, COM1, COM2, partDiameter):
    COMX = COM1[0] 
    COMY = COM1[1]
    COMZ = COM1[2] 
    # return numpy.array([choice([randint( int(COMX - ((LD2 + LD1)/2 + partDiameter)), int(COMX - LD1/2)), randint( int(COMX + LD1/2), int(COMX + (LD1 + LD2)/2 + partDiameter) )]),
                        # choice([randint( int(COMY - ((LD2 + LD1)/2 + partDiameter)), int(COMY - LD1/2)), randint( int(COMY + LD1/2), int(COMY + (LD1 + LD2)/2 + partDiameter) )]),
                        # choice([randint( int(COMZ - ((LD2 + LD1)/2 + partDiameter)), int(COMZ - LD1/2)), randint( int(COMZ + LD1/2), int(COMZ + (LD1 + LD2)/2 + partDiameter) )])])
    return numpy.array([choice([randint( int(COMX - (LD2 + LD1/2 + partDiameter)), int(COMX - LD1/2)), randint( int(COMX + LD1/2), int(COMX + LD1/2 + LD2 + partDiameter) )]),
                        choice([randint( int(COMY - (LD2 + LD1/2 + partDiameter)), int(COMY - LD1/2)), randint( int(COMY + LD1/2), int(COMY + LD1/2 + LD2 + partDiameter) )]),
                        choice([randint( int(COMZ - (LD2 + LD1/2 + partDiameter)), int(COMZ - LD1/2)), randint( int(COMZ + LD1/2), int(COMZ + LD1/2 + LD2 + partDiameter) )])])



# def cluster_testing(agg1, agg2, partDiameter):
#     """ Returns the positions of agg2 when agg1 and agg2 are touching
#     """
#     agg2_temp = translate_aggregate(agg2, random_point_generator(calculate_LD(agg1), calculate_LD(agg2), calculate_COM(agg1), calculate_COM(agg2), partDiameter))
#     agg2_temp = random_rotate_aggregate(agg2_temp)
#     check = 1
#     while check == 1:
#         agg2_temp = translate_aggregate(agg2_temp, numpy.array((calculate_COM(agg1)-calculate_COM(agg2_temp))*0.1))
#         result1, result2 = test_collision(agg1, agg2_temp, partDiameter)
#         if result1 == 2:
#             return calculate_COM(agg2_temp)
#             break 


def cluster_testing_dist(agg1, agg2, partDiameter):
    """ Returns the absolute distance between com of agg1 and com of agg2
    """
    agg2_temp = translate_aggregate(agg2, random_point_generator(calculate_LD(agg1), calculate_LD(agg2), calculate_COM(agg1), calculate_COM(agg2), partDiameter))
    agg2_temp = random_rotate_aggregate(agg2_temp)

    check = 1
    while check == 1:
        agg2_temp = translate_aggregate(agg2_temp, numpy.array((calculate_COM(agg1)-calculate_COM(agg2_temp))*0.01))
        check, index = test_collision(agg1, agg2_temp, partDiameter)
        """ Index from this part is not valid! Function returns '99' before collision happens.
        """
        if (check == 2):
            # print(index)
            return numpy.linalg.norm(calculate_COM(agg1) - calculate_COM(agg2_temp)), numpy.linalg.norm(calculate_COM(agg1) - agg2_temp[:,index])
            # return numpy.linalg.norm(calculate_COM(agg1) - agg2_temp[0:3,index])
            break 




def  launch_vpython(agglomerate, partDiameter):

    display()
    scene2 = display(title = 'Aggregate topology',
                x = 200, y = 22, width = 800, height = 600,
                center = (0,0,0), background = (1,1,1))

    # draw x-y-z planes: 
    mybox = box(pos=(0., 0., 0.), axis = (1,0,0), length = 0.01,
          height = 20, width = 20, color = color.red, opacity = 0.1)
    mybox = box(pos=(0., 0., 0.), axis = (0,1,0), length = 0.01,
          height = 20, width = 20, color = color.green, opacity = 0.1)
    mybox = box(pos=(0., 0., 0.), axis = (0,0,1), length = 0.01,
          height = 20, width = 20, color = color.blue, opacity = 0.1)


    for i in xrange(agglomerate.shape[1]):
        ball = sphere(pos = vector(agglomerate[0,i],
                             agglomerate[1,i],
                             agglomerate[2,i]), radius = 0.5*partDiameter, opacity=0.4, color = color.red)


def perform_experiment(infile, outfile, number_of_shoot):
   
    print (infile, outfile)
    partDiameter = 1.0

    data1 = numpy.loadtxt(infile)                        
    data2 = data1
    # data2 = numpy.loadtxt("agg133t.dat")                        
    # here input file contains 5 columns: cluster ID, particle ID, x-, z- and y- position of particles,
    # therefore is the [:, 2:5]
    # Also, input data are written in rows, to transpose it into columns is used transform_input_data() 
    agg0 = transform_input_data(numpy.array(data1[:,2:5])) 
    agg1 = transform_input_data(numpy.array(data2[:,2:5])) 

    outfile = open(outfile, 'a')
    # launch_vpython(agg0, partDiameter)
    print ("Number of shoot:")
    for x in xrange(1,number_of_shoot):
        print(x)
        k, g = cluster_testing_dist(agg0,agg1, partDiameter)
        # t = cluster_testing(agg0,agg1, partDiameter)
        # print('{0:2d} {1:3f}'.format(x, k))
        outfile.write('{0:2d} {1:3f} {2:4f}\n'.format(x, k, g))
        # outfile.write('{0:2d} {1:3f}\n'.format(x, g))
    outfile.close()


# print("----------------------------------------------------------------------------------------------------")
# print("------------------------------------>     MAIN PART  <----------------------------------------------")
# print("----------------------------------------------------------------------------------------------------")

perform_experiment(args.input, args.output, 100)
print ("END")

# print("----------------------------------------------------------------------------------------------------")
# print("------------------------------------>  END OF THE FILE <--------------------------------------------")
# print("----------------------------------------------------------------------------------------------------")
