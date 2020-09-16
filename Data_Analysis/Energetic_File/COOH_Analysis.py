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
        nAtoms = int(p.readline().split()[0]) + 4 # specifically for OCOH system
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
    
def get_centerofmass_OCOH(c_coor, o1_coor, o2_coor, h_coor):
    """Return the center of mass of CO"""
    c_mass = 12.0107 
    o_mass = 15.999
    h_mass =  1.0078
    mass_sum = c_mass + o_mass * 2 + h_mass
    x_com = (c_mass * c_coor[0] + o_mass * o1_coor[0] + 
             o_mass * o2_coor[0] + h_mass * h_coor[0]) / mass_sum
    y_com = (c_mass * c_coor[1] + o_mass * o1_coor[1] + 
             o_mass * o2_coor[1] + h_mass * h_coor[1]) / mass_sum
    z_com = (c_mass * c_coor[2] + o_mass * o1_coor[2] + 
             o_mass * o2_coor[2] + h_mass * h_coor[2]) / mass_sum
    return [x_com, y_com, z_com]

def get_dist(atom_1, atom_2):
    """Return the distance between atom 1 and atom 2"""
    dist = np.linalg.norm(atom_1 - atom_2)
    return dist

def get_discriptor(AuOCOH_array):
    """Assemble the discriptors"""
    center_Au = AuOCOH_array[0]
    C = AuOCOH_array[-4]
    O1 = AuOCOH_array[-3]
    O2 = AuOCOH_array[-2]
    H = AuOCOH_array[-1]
    com_OCOH = get_centerofmass_OCOH(C, O1, O2, H)
    dis_AuC = get_dist(center_Au, C)
    dis_com = get_dist(center_Au, com_OCOH)
    return dis_AuC, dis_com

##############################################################################
##############################################################################
#Read vacuum data
vacuum_map = {}

f_read_vacuum = open('Vacuum_AuCOOH.csv', 'r')
f_read_vacuum.readline()
for i in range(1059):
    line = f_read_vacuum.readline()
    index = line.split()[0]
    coorfination = int(line.split()[1])
    Ead = float(line.split()[2])
    vacuum_map[index] = (coorfination, Ead)
f_read_vacuum.close()

f_write = open('Final_AuCOOH_Summary.csv', 'w')
f_write.writelines('#' * 79 + '\n')
f_write.writelines('Defintion of the variables\n\
  index: ID of structure\n\
  coordiantion: number of the first neigbor of the center Au\n\
  Ead: adsorption energy\n\
  AuC: distance between center Au and C\n\
  com_dis: distance between center Au and center of mass of OCOH\n')
f_write.writelines('#' * 79 + '\n\n')
f_write.writelines('index coordination Ead_vacuum Ead_reqm AuC_vacuum AuC_reqm\
com_dis_vacuum com_dis_reqm\n')
f_read_reqm = open('ReQM_AuCOOH.csv', 'r')
f_read_reqm.readline()
for i in range(1059):
    line = f_read_reqm.readline()
    index = line.split()[0].split('_')[2]
    Ead_reqm = float(line.split()[4])
    n1, AuOCOH_array_vacuum = getCoord_CONTCAR('../AuCOOH_Vacuum/Vacuum_AuCOOH_{}'.format(index), 'D')
    n2, AuOCOH_array_reqm = getCoord_CONTCAR('../AuCOOH_ReQM/ReQM_AuCOOH_{}'.format(index), 'C')
    dis1_vacuum, dis2_vacuum = get_discriptor(AuOCOH_array_vacuum)
    dis1_reqm, dis2_reqm = get_discriptor(AuOCOH_array_reqm)
    coordination = vacuum_map[index][0]
    Ead_vacuum = vacuum_map[index][1]
    f_write.writelines('{:>8s} {:3d}  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f} \
{:8.4f}  {:8.4f}\n'.format(index, coordination, Ead_vacuum,Ead_reqm, dis1_vacuum, \
                            dis1_reqm, dis2_vacuum, dis2_reqm))
f_write.close()
f_read_reqm.close()

    