import os,sys
import numpy as np
import matplotlib.pyplot as plt

def getCoord_CONTCAR(CONTCAR, flag):
    '''
    Return the cartesian coordinate array from the CONTCAR
    Input:
        CONTCAR: Coordiate file
        flag: cartesian type in Input file
    Output:
        Number of atoms, cartesian array of the nanocluster
    '''
    with open(CONTCAR,'r') as p:
        lattice = np.zeros((3,3))
        p.readline()
        p.readline()
        lattice[0,:] = np.array(p.readline().split(), dtype=float)
        lattice[1,:] = np.array(p.readline().split(), dtype=float)
        lattice[2,:] = np.array(p.readline().split(), dtype=float)
        p.readline()
        nAtoms = int(p.readline().split()[0]) + 2 # specifically for CO system
        p.readline()
        if flag == 'D': #Direct Coordinates
            p.readline()
            R = np.zeros((nAtoms,3))
            for i in range(nAtoms):
                R[i] = np.array(p.readline().split()[:3],dtype=float)
            return nAtoms, R.dot(lattice)
    
        if flag == 'C': #Cartesian Coordiates
            R = np.zeros((nAtoms,3))
            for i in range(nAtoms):
                R[i] = np.array(p.readline().split()[:3],dtype=float)
            return nAtoms, R
    
def get_centerofmass_CO(c_coor, o_coor):
    """Return the center of mass of CO"""
    c_mass = 12.0107 
    o_mass = 15.999
    x_com = (c_mass * c_coor[0] + o_mass * o_coor[0]) / (c_mass + o_mass)
    y_com = (c_mass * c_coor[1] + o_mass * o_coor[1]) / (c_mass + o_mass)
    z_com = (c_mass * c_coor[2] + o_mass * o_coor[2]) / (c_mass + o_mass)
    return [x_com, y_com, z_com]

def get_dist(atom_1, atom_2):
    """Return the distance between atom 1 and atom 2"""
    dist = np.linalg.norm(atom_1 - atom_2)
    return dist

def get_angle(atom_1, atom_2, atom_3):
    """Return the angle between atom_1-atom_2-atom_3"""
    vec_12= atom_2 - atom_1
    vec_13 = atom_3 - atom_1
    cos = np.dot(vec_12, vec_13)/np.linalg.norm(vec_12)/np.linalg.norm(vec_13)
    angle = np.arccos(cos)*180/np.pi
    return angle

def get_discriptor(AuCO_array):
    """Assemble the discriptors"""
    center_Au = AuCO_array[0]
    C = AuCO_array[-2]
    O = AuCO_array[-1]
    com_CO = get_centerofmass_CO(C, O)
    dis_AuC = get_dist(center_Au, C)
    dis_com = get_dist(center_Au, com_CO)
    angle = get_angle(C, center_Au, O)
    return dis_AuC, dis_com, angle

##############################################################################
##############################################################################
#Read vacuum data
vacuum_map = {}
f_read_vacuum = open('Vacuum_AuCO.csv', 'r')
f_read_vacuum.readline()
for i in range(1385):
    line = f_read_vacuum.readline()
    index = line.split()[0]
    coorfination = int(line.split()[1])
    Ead = float(line.split()[2])
    vacuum_map[index] = (coorfination, Ead)
f_read_vacuum.close()

#read reqm data and write summery file
f_write = open('Final_AuCO_Summary.csv', 'w')
f_write.writelines('#' * 79 + '\n')
f_write.writelines('Defintion of the variables\n\
  index: ID of structure\n\
  coordiantion: number of the first neigbor of the center Au\n\
  Ead: adsorption energy\n\
  AuC: distance between center Au and C\n\
  com_dis: distance between center Au and center of mass of CO\n\
  angle: angle between center Au, C and O\n')
f_write.writelines('#' * 79 + '\n\n')
f_write.writelines('index coordination Ead_vacuum Ead_reqm AuC_vacuum AuC_reqm\
com_dis_vacuum com_dis_reqm angle_vacuum angle_reqm\n')
f_read_reqm = open('ReQM_AuCO.csv', 'r')
f_read_reqm.readline()
for i in range(1385):
    line = f_read_reqm.readline()
    index = line.split()[0].split('_')[2]
    Ead_reqm = float(line.split()[4])
    n1, AuCO_array_vacuum = getCoord_CONTCAR('../AuCO_Vacuum/Vacuum_AuCO_{}'.format(index), 'D')
    n2, AuCO_array_reqm = getCoord_CONTCAR('../AuCO_ReQM/ReQM_AuCO_{}'.format(index), 'C')
    dis1_vacuum, dis2_vacuum, angle_vacuum = get_discriptor(AuCO_array_vacuum)
    dis1_reqm, dis2_reqm, angle_reqm = get_discriptor(AuCO_array_reqm)
    coordination = vacuum_map[index][0]
    Ead_vacuum = vacuum_map[index][1]
    f_write.writelines('{:>8s} {:3d}  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f} \
{:8.4f}  {:8.4f}  {:8.2f}  {:8.2f}\n'.format(index, coordination, Ead_vacuum,\
                                             Ead_reqm, dis1_vacuum, dis1_reqm, \
                                             dis2_vacuum, dis2_reqm, angle_vacuum,\
                                             angle_reqm))
f_write.close()
f_read_reqm.close()
    