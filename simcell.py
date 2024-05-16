#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 12:53:32 2017
Current version Mon Oct 23 14:34:11 2023

@author: Vadim Sultanov
"""

import sys
import copy
import time
import numpy as np
import ptable
from importlib import import_module
from copy import deepcopy
import io
import re

class c_general:
    "Just a structure with fields defined by constructor arguments"
    def __init__(self, **kwargs):
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

week_abbr = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] # ????
month_abbr = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def cartesian_product(list_of_lists):
    N = len(list_of_lists)
    lengths = [len(l) for l in list_of_lists]
    counters = [0] * N
    while counters[0] < lengths[0]:
        yield [list_of_lists[i][counters[i]] for i in range(N)]
        counters[-1] += 1
        for i in range(N-1, 0, -1):
            if counters[i] == lengths[i]:
                counters[i] = 0
                counters[i-1] += 1

def rotate3d(phi, a, p, c = (0., 0., 0.)):
    """Rotates point p around axis a (must be a unit vector)
    with origin c by angle phi"""

    a_x, a_y, a_z = a
    p_x, p_y, p_z = p
    c_x, c_y, c_z = c

    #subtracting origin
    v_x = p_x - c_x; v_y = p_y - c_y; v_z = p_z - c_z

    # forming a rotation quaternion
    sinhalfphi = np.sin(phi/2.)
    w = np.cos(phi/2.)
    x = sinhalfphi * a_x
    y = sinhalfphi * a_y
    z = sinhalfphi * a_z

    # multiplying the vector by inverse rotation quaternion (norm of the quaternion = 1)
    q_x = v_x * w + v_z * y - v_y * z
    q_y = v_y * w + v_x * z - v_z * x
    q_z = v_z * w + v_y * x - v_x * y
    q_w = v_x * x + v_y * y + v_z * z

    # multiplying the two formers
    ret_x = w * q_x + x * q_w + y * q_z - z * q_y
    ret_y = w * q_y + y * q_w + z * q_x - x * q_z
    ret_z = w * q_z + z * q_w + x * q_y - y * q_x

    #re-adding origin
    ret_x += c_x; ret_y += c_y; ret_z += c_z

    return (ret_x, ret_y, ret_z)

def orient(r, m):
    """For a group of point masses m located at coordinates r
    return new coordinates in the coordinate system of
    main axes of inertia. Axis corresponding to maximum main
    moment of inertia becomes x, and its minimum counterpart
    becomes z. Input values r and m are not altered."""

    M = np.sum(m)
    N = len(m)
    com = np.sum(r * m.reshape(N, 1) / M, axis = 0)
    rcom = r - com
    mxx = np.sum(m * rcom[:, 0] * rcom[:, 0])
    mxy = np.sum(m * rcom[:, 0] * rcom[:, 1])
    mxz = np.sum(m * rcom[:, 0] * rcom[:, 2])
    myy = np.sum(m * rcom[:, 1] * rcom[:, 1])
    myz = np.sum(m * rcom[:, 1] * rcom[:, 2])
    mzz = np.sum(m * rcom[:, 2] * rcom[:, 2])
    # tensor of inertia
    I = [[ myy + mzz, -mxy, -mxz], \
         [-mxy,  mxx + mzz, -myz], \
         [-mxz, -myz,  mxx + myy]]
    mainmom, axes = np.linalg.eig(I)
    detI = np.linalg.det(axes)
    iz = np.argmin(mainmom)
    ix = np.argmax(mainmom)
    if iz == ix:
        ix = 0; iy = 1; iz = 2
    else:
        iy = 3 - (ix + iz)
    xaxis = axes[:, ix]
    yaxis = axes[:, iy]
    zaxis = axes[:, iz]
    n = 0
    if ix != 0:
        n += 1
    if iy != 1:
        n += 1
    if iz != 2:
        n += 1
    if n == 2:
        detI *= -1
    if detI < 0:
        xaxis *= -1
    ret = np.empty_like(rcom)
    ret[:, 0] = np.inner(rcom, xaxis)
    ret[:, 1] = np.inner(rcom, yaxis)
    ret[:, 2] = np.inner(rcom, zaxis)
    return ret, xaxis, yaxis, zaxis


def zpoint(p0, l, p1, a, p2, d):
    """Calculates coordinates of atom defined by line of z-matrix (pnew).
    p0, p1, p2 are coordinates of three reference atoms (array-like).
    l is distance pnew_p0; a is angle pnew_p0_p1 (degrees);
    d is dihedral angle pnew_p0_p1_p2 (degrees)."""

#    p0 = np.array(p0)
#    p1 = np.array(p1)
#    p2 = np.array(p2)

    # building a new frame (basis)

    z = p0 - p1
    z /= np.linalg.norm(z)

    x = p2 - p1
    x -= z * np.dot(z, x)
    x /= np.linalg.norm(x)

    y = np.cross(z, x)

    th = a * np.pi / 180.   # polar angle
    ph = d * np.pi / 180.   # azimuthal angle
    sin_th = np.sin(th); cos_th = np.cos(th)
    sin_ph = np.sin(ph); cos_ph = np.cos(ph)

    ret = p0 + l * (x * sin_th * cos_ph + y * sin_th * sin_ph - z * cos_th)
    return ret

def dihedral(pl0, pm0, pm1, pl1):
    """Calculate the measure of dihedral angle (rad)
    formed by points (or arrays of points) pl0, pm0, pm1, pl1 (numpy arrays)"""

    if pl0.ndim == 1:
        axis = pm1 - pm0
        axis /= np.linalg.norm(axis)
        rad1 = (pl0 - pm0) - axis * np.sum(axis * (pl0 - pm0))
        rad1 /= np.sqrt(np.sum(rad1**2))
        rad2 = (pl1 - pm1) - axis * np.sum(axis * (pl1 - pm1))
        rad2 /= np.sqrt(np.sum(rad2**2))

        ret = np.arccos(np.sum(rad1*rad2))

        if np.linalg.det([axis, rad1, rad2]) < 0.0:
            ret = 2.*np.pi - ret

    elif pl0.ndim == 2:
        N = pl0.shape[0]
        axis = pm1 - pm0
        axis /= np.sqrt(np.sum(axis**2, 1)).reshape((N, 1))
        rad1 = (pl0 - pm0) - axis * np.sum(axis * (pl0 - pm0), 1).reshape((N, 1))
        rad1 /= np.sqrt(np.sum(rad1**2, 1)).reshape((N, 1))
        rad2 = (pl1 - pm1) - axis * np.sum(axis * (pl1 - pm1), 1).reshape((N, 1))
        rad2 /= np.sqrt(np.sum(rad2**2, 1)).reshape((N, 1))

        ret = np.arccos(np.sum(rad1*rad2, 1))

        for i in range(N):
            if np.linalg.det([axis[i, :], rad1[i, :], rad2[i, :]]) < 0.0:
                ret[i] = 2.*np.pi - ret[i]

    elif pl0.ndim < 1:
        return None

    else:
        butone = pl0.shape
        butone[-1] = 1
        lastdim = pl0.ndim - 1
        axis = pm1 - pm0
        axisnorm = np.sqrt(np.sum(axis**2, lastdim))
        axis /= axisnorm.reshape(butone)
        rad1 = (pl0 - pm0) - axis * np.sum(axis * (pl0 - pm0), lastdim).reshape(butone)
        rad1 /= np.sqrt(np.sum(rad1**2, lastdim)).reshape(butone)
        rad2 = (pl1 - pm1) - axis * np.sum(axis * (pl1 - pm1), lastdim).reshape(butone)
        rad2 /= np.sqrt(np.sum(rad2**2, lastdim)).reshape(butone)

        ret = np.arccos(np.sum(rad1*rad2, lastdim))

        it = np.nditer(axisnorm, flags=['multi_index'])
        while not it.finished:
            mi = it.multi_index
            if np.linalg.det([axis[mi], rad1[mi], rad2[mi]]) < 0.0:
                ret[mi] = 2.*np.pi - ret[mi]
            it.iternext()

    return ret


def room_test_pbc(r, hsigma, directory, size):
    """Check whether all atoms located at r[i] with vdw radii
    hsigma[i] are distant enough from atoms in directory
    (previously added by this same function).
    If yes, also adds these atoms to directory.
    directory should be initialized before first call of
    room_test_pbc as np.empty((Nx, Ny, Nz), dtype = object).
    Distances are calculated as minimum distances within
    a periodic box with dimensions size = np.array([Lx, Ly, Lz]).
    L_/N_ must be >= 2 * maximum vdw radius in the system"""

    Ntiles = np.array(directory.shape, dtype = int)
    tile_size = size / Ntiles
    N = len(r)
    rs = np.empty((N, 4))
    rs[:, :3] = r % size.reshape((1, 3))
    rs[:, 3] = hsigma
    indices = (rs[:, :3] / tile_size.reshape((1, 3))).astype(int)
    trial_dir = {}
    for i in range(N):
        key = tuple(indices[i])
        if key not in trial_dir:
            trial_dir[key] = []
        trial_dir[key].append(rs[i])

    for key in trial_dir:
        trial_dir[key] = rs_new = np.array(trial_dir[key])
        N_new = rs_new.shape[0]
        kx, ky, kz = key
        for ix in [kx-1, kx, kx+1]:
            for iy in [ky-1, ky, ky+1]:
                for iz in [kz-1, kz, kz+1]:
                    ic = [ix, iy, iz]
                    correction = np.zeros(3)
                    for j in range(3):
                        if ic[j] < 0:
                            ic[j] += Ntiles[j]
                            correction[j] = size[j]
                        if ic[j] >= Ntiles[j]:
                            ic[j] -= Ntiles[j]
                            correction[j] = -size[j]
                    ix_, iy_, iz_ = ic
                    rs_old = directory[ix_, iy_, iz_]
                    if rs_old is None:
                        continue
                    N_old = rs_old.shape[0]
                    dr = np.empty((N_old, N_new, 3))
                    dr[:, :, 0] = np.subtract.outer(rs_old[:, 0], rs_new[:, 0])
                    dr[:, :, 1] = np.subtract.outer(rs_old[:, 1], rs_new[:, 1])
                    dr[:, :, 2] = np.subtract.outer(rs_old[:, 2], rs_new[:, 2])
                    sigma = np.add.outer(rs_old[:, 3], rs_new[:, 3])
                    dr += correction.reshape((1, 1, 3))
                    distance = np.linalg.norm(dr, axis = 2)
                    if not np.all(distance >= sigma):
                        return False

    for key in trial_dir:
        rs_old = directory[key]
        if rs_old is None:
            directory[key] = trial_dir[key]
        else:
            directory[key] = np.concatenate((rs_old, trial_dir[key]))

    return True

def room_test_nonperiodic(r, hsigma, directory, sigma_max = None):
    """Check whether all atoms located at r[i] with vdw radii
    hsigma[i] are distant enough from atoms in directory
    (previously added by this same function).
    If yes, also adds these atoms to directory.
    directory should be initialized before first call of
    room_test_pbc as {}.
    sigma_max = 2 * maximum vdw radius in the system"""

    if sigma_max is None:
        sigma_max  = np.max(hsigma) * 2
    N = len(r)
    rs = np.concatenate((r, hsigma.reshape((N, 1))), axis = 1)
    indices = (r / sigma_max).astype(int)
    trial_dir = {}
    for i in range(N):
        key = tuple(indices[i])
        if key not in trial_dir:
            trial_dir[key] = []
        trial_dir[key].append(rs[i])

    for key in trial_dir:
        trial_dir[key] = rs_new = np.array(trial_dir[key])
        N_new = rs_new.shape[0]
        kx, ky, kz = key
        for ix in [kx-1, kx, kx+1]:
            for iy in [ky-1, ky, ky+1]:
                for iz in [kz-1, kz, kz+1]:
                    rs_old = directory.get((ix, iy, iz))
                    if rs_old is None:
                        continue
                    N_old = rs_old.shape[0]
                    dr = np.empty((N_old, N_new, 3))
                    dr[:, :, 0] = np.subtract.outer(rs_old[:, 0], rs_new[:, 0])
                    dr[:, :, 1] = np.subtract.outer(rs_old[:, 1], rs_new[:, 1])
                    dr[:, :, 2] = np.subtract.outer(rs_old[:, 2], rs_new[:, 2])
                    sigma = np.add.outer(rs_old[:, 3], rs_new[:, 3])
                    distance = np.linalg.norm(dr, axis = 2)
                    if not np.all(distance >= sigma):
                        return False

    for key in trial_dir:
        rs_old = directory.get(key)
        if rs_old is None:
            directory[key] = trial_dir[key]
        else:
            directory[key] = np.concatenate((rs_old, trial_dir[key]))

    return True

def self_room_test(r, hsigma, size):
    """Check whether all atoms located at r[i] with vdw radii
    hsigma[i] are distant enough from their replicas
    in periodic box with dimensions size = np.array([Lx, Ly, Lz])"""

    sigma_max = np.max(hsigma) * 2
    directory = {}
    room_test_nonperiodic(r, hsigma, directory, sigma_max)
    for i in range(3):
        replica = np.copy(r)
        replica[:, i] += size[i]
        if not room_test_nonperiodic(replica, hsigma, directory, sigma_max):
            return False
    return True

class random_polar_angle:
    """To obtain random points uniformly distributed over a sphere
    one should take uniformly distributed azimuthal angle
    and polar angle distributed with account of length of
    'circle of latitude'"""

    def __init__(self, Nbins = 360):
        "Higher Nbins provide higher precision of the distribution"
        self.Nbins = Nbins
        th_v = np.linspace(0, 1, self.Nbins * 100 + 1)
        th_a = (th_v - np.sin(2*np.pi*th_v)/2/np.pi) * self.Nbins
        self.th_y = np.zeros(self.Nbins + 1)
        for i in range(self.Nbins + 1):
            self.th_y[i] = np.argmin(np.abs(th_a - i)) / self.Nbins / 100
    def __call__(self):
        rval = np.random.rand() * self.Nbins
        i = int(rval)
        return np.pi * (self.th_y[i] + (rval - i) * (self.th_y[i+1] - self.th_y[i]))

class c_dump:
    """Iterator & list-like object representing LAMMPS dump file
    Returns one frame as list of lines"""

    def __init__(self, dumpname):

        self.dumpname = dumpname
        self.file = open(self.dumpname, 'r')
        self.oframe = 0
        self.oline = 0
        self.framestarts = []

        frame = self.__next__()
        self.file.seek(0)

        self.oframe = 0
        self.oline = 0
        self.framestarts = []

        splitdata = [l.split() for l in frame[9:]]
        for record in splitdata:
            record[0] = int(record[0])

        self.first_timestep = int(frame[1])
        self.Natoms = int(frame[3])
        splitl4 = frame[4][:-1].split()
        if len(splitl4) > 3:
            self.periodicity = tuple(splitl4[3:])
        splitl8 = frame[8][:-1].split()
        if len(splitl8) > 2:
            self.signature = tuple(splitl8[2:])
        else:
            self.signature = ('id', 'type', 'x', 'y', 'z', 'ix', 'iy', 'iz')
        if self.signature[0] != 'id':
            raise Exception('First element of dump record is not "id"')
        atomlist = [record[0] for record in splitdata]
        atomlist.sort()
        atomlist.insert(0, 0)
        self.atomlist = atomlist
        self.is_sparse = (atomlist[-1] != self.Natoms)
        if self.is_sparse:
            self.dict = {b: a for (a, b) in enumerate(atomlist)}
        self.x_noimageinfo = ('x' in self.signature or 'xs' in self.signature) and ('ix' not in self.signature)
        self.y_noimageinfo = ('y' in self.signature or 'ys' in self.signature) and ('iy' not in self.signature)
        self.z_noimageinfo = ('z' in self.signature or 'zs' in self.signature) and ('iz' not in self.signature)


    def __del__(self):

        if hasattr(self, 'file'):
            self.file.close()


    def __iter__(self):

        if hasattr(self, 'file'):
            self.file.seek(0)
        else:
            self.file = open(self.dumpname, 'r')
        self.oframe = 0
        self.oline = 0
        return self


    def __next__(self):

        if not hasattr(self, 'file'):
            raise StopIteration()
        firstline = self.oline
        frame = []
        ln = ''
        try:
            while not 'ITEM: NUMBER OF ATOMS' in ln:
                ln = self.file.__next__(); self.oline += 1
                frame.append(ln)
            ln = self.file.__next__(); self.oline += 1
            frame.append(ln)
            Natoms = int(ln)
            while not 'ITEM: ATOMS' in ln:
                ln = self.file.__next__(); self.oline += 1
                frame.append(ln)
            for i in range(Natoms):
                ln = self.file.__next__(); self.oline += 1
                frame.append(ln)
        except:
            self.file.close()
            delattr(self, 'file')
            self.Nframes = len(self.framestarts)
            raise StopIteration()

        if len(self.framestarts) == self.oframe:
            self.framestarts.append(firstline)
        self.oframe += 1

        return frame


    def __len__(self):

        if not hasattr(self, 'Nframes'):
            for frame in self:
                pass
        return self.Nframes


    def __getitem__(self, key):

        if not hasattr(self, 'file'):
            self.file = open(self.dumpname, 'r')
            self.oframe = 0
            self.oline = 0

        if hasattr(self, 'Nframes'):
            if key < -self.Nframes or key >= self.Nframes:
                raise IndexError()
        elif key < 0:
            try:
                while True:
                    frame = self.__next__()
            except: pass
            if key == -1 and self.Nframes > 0:
                return frame
            else:
                return self[key]

        if key < 0:
            key += self.Nframes
        if self.oframe > key:
            self.file.seek(0)
            self.oframe = 0
            self.oline = 0
        try:
            while self.oframe <= key:
                frame = self.__next__()
        except:
            raise IndexError()

        return frame

    def digest(self, frame):
        """
        Return a frame 'digest' - the structure with fields:
            timestep
            xlo, xhi, ylo, yhi, zlo, zhi
            xy, xz, yz (if triclinic)
            xsize, ysize, zsize
            is_sparse
            atomlist (if sparse)
            array (raw array of all values contained in dump frame table)
            r (3*N matrix of unwrapped atoms coordinates)
            v (3*N matrix of atoms velocities)
        """

        ret = c_general()

        splitdata = [l.split() for l in frame[9:]]
        for record in splitdata:
            record[0] = int(record[0])

        ret.timestep = int(frame[1])
        fs = [float(t) for t in frame[5].split()]
        ret.xlo = fs[0]; ret.xhi = fs[1]
        if len(fs) > 2: ret.xy = fs[2]
        fs = [float(t) for t in frame[6].split()]
        ret.ylo = fs[0]; ret.yhi = fs[1]
        if len(fs) > 2: ret.xz = fs[2]
        fs = [float(t) for t in frame[7].split()]
        ret.zlo = fs[0]; ret.zhi = fs[1]
        if len(fs) > 2: ret.yz = fs[2]
        ret.xsize = ret.xhi - ret.xlo
        ret.ysize = ret.yhi - ret.ylo
        ret.zsize = ret.zhi - ret.zlo
        ret.is_sparse = self.is_sparse

        lenarray = self.Natoms + 1
        # if not self.is_sparse:
        #     lenarray += 1
        is_v = ('vx' in self.signature) and ('vy' in self.signature) and ('vz' in self.signature)

        ret.array = np.zeros( (lenarray, len(self.signature)) )
        ret.r = np.zeros( (lenarray, 3) )
        if is_v:
            ret.v = np.zeros( (lenarray, 3) )

        if self.is_sparse:
            ret.atomlist = self.atomlist
            for record in splitdata:
                i = self.dict[record[0]]
                ret.array[i, :] = [float(a) for a in record]
        else:
            for record in splitdata:
                i = record[0]
                ret.array[i, :] = [float(a) for a in record]

        if 'xu' in self.signature:
            ret.r[:, 0] = ret.array[:, self.signature.index('xu')]
        elif 'xsu' in self.signature:
            ret.r[:, 0] = ret.array[:, self.signature.index('xsu')] * ret.xsize
        elif 'x' in self.signature:
            ret.r[:, 0] = ret.array[:, self.signature.index('x')]
        elif 'xs' in self.signature:
            ret.r[:, 0] = ret.array[:, self.signature.index('xs')] * ret.xsize
        if ('x' in self.signature or 'xs' in self.signature) and 'ix' in self.signature:
                ret.r[:, 0] += ret.array[:, self.signature.index('ix')] * ret.xsize

        if 'yu' in self.signature:
            ret.r[:, 1] = ret.array[:, self.signature.index('yu')]
        elif 'ysu' in self.signature:
            ret.r[:, 1] = ret.array[:, self.signature.index('ysu')] * ret.ysize
        elif 'y' in self.signature:
            ret.r[:, 1] = ret.array[:, self.signature.index('y')]
        elif 'ys' in self.signature:
            ret.r[:, 1] = ret.array[:, self.signature.index('ys')] * ret.ysize
        if ('y' in self.signature or 'ys' in self.signature) and 'iy' in self.signature:
                ret.r[:, 1] += ret.array[:, self.signature.index('iy')] * ret.ysize

        if 'zu' in self.signature:
            ret.r[:, 2] = ret.array[:, self.signature.index('zu')]
        elif 'zsu' in self.signature:
            ret.r[:, 2] = ret.array[:, self.signature.index('zsu')] * ret.zsize
        elif 'z' in self.signature:
            ret.r[:, 2] = ret.array[:, self.signature.index('z')]
        elif 'zs' in self.signature:
            ret.r[:, 2] = ret.array[:, self.signature.index('zs')] * ret.zsize
        if ('z' in self.signature or 'zs' in self.signature) and 'iz' in self.signature:
                ret.r[:, 2] += ret.array[:, self.signature.index('iz')] * ret.zsize

        if is_v:
            ret.v[:, 0] = ret.array[:, self.signature.index('vx')]
            ret.v[:, 1] = ret.array[:, self.signature.index('vy')]
            ret.v[:, 2] = ret.array[:, self.signature.index('vz')]

        return ret


### c_cell attributes
# ====================
#
#=== Static data (all inner items have type str).
#
# count_members    list of c_cell count members
# chapters         list of LAMMPS data file chapters names
# link_chapters    sublist of the previous
# coef_chapters    dict {LAMMPS data file Coeff chapter name: corresponding c_cell count member}
# counts           LAMMPS data file counts (atoms, bonds, etc.)
#
#== Cell geometry. float.
#== Compulsory
# xhi
# xlo
# yhi
# ylo
# zhi
# zlo
#== Present if is_tilted
# xy
# xz
# yz
#
#== LAMMPS data file chapter information. Compulsory.
#
# chapter_present    dict {str: bool} True if chapter is present in data file
# chapter_comment    dict {str: str} Comment to chapter; '' if no comment
#
#== Description in the top of data file. str. Compulsory.
#
# comment
#
#== c_cell characteristics. bool. Compulsory.
#
# is_empty
# is_tilted
#
#== Count members. int. Compulsory.
#
# N
# Nbonds
# Nangles
# Ndihedrals
# Nimpropers
#
# Nmols
#
# Ntypes
# Nbtypes
# Natypes
# Ndtypes
# Nitypes
#
#== Atom properties. numpy array. Present if not is_empty (except v).
#
# omol    shape = (N+1,)    dtype = int      Ordinal number of the enclosing molecule
# q       shape = (N+1,)    dtype = float    Partial charge of the atom
# r       shape = (N+1, 3)  dtype = float    Atom coordinates
# v       shape = (N+1, 3)  dtype = float    Atom velocities. Present if chapter_present['Velocities']
# t       shape = (N+1,)    dtype = np.uint8 Atom type
# text    shape = (N+1,)    dtype = object   Comment to the atom
#
#== Links (involved atoms). numpy array, dtype = int. Optional; present if chapter_present['Link'] is true.
#
# bond        shape = (Nbonds + 1, 2)
# angle       shape = (Nangles + 1, 3)
# dihedral    shape = (Ndihedrals + 1, 4)
# improper    shape = (Nimpropers + 1, 4)
#
#== Link types. numpy array, shape = (Nlink + 1,), dtype = np.uint8. Optional; present if chapter_present['Link'] is true.
#
# bond_t
# angle_t
# dihedral_t
# improper_t
#
#== Bonds orders. numpy array.
#
# bond_order    shape = (Nbonds+1,) dtype = np.uint8
#
#== Molecule names. numpy array.
#
# mol_name    shape = (Nmols+1,)    dtype = object
#
#== Ordinal number of the first item enclosed in the molecule.
#== numpy array. dtype = int shape = (Nmols+2,).
#== The last value in the array is Nitem+1.
#== Present if not is_empty.
#
# mol_1st     Atoms
# mol_b1st    Bonds
# mol_a1st    Angles
# mol_d1st    Dihedrals
# mol_i1st    Impropers
#
#== Atom type properties. numpy array. shape = (Ntypes+1,). Present if chapter_present['Masses']
#
# type_elm    dtype = 'U2'   Element symbol
# type_ff     dtype = 'U8'   Force field type
# type_m      dtype = float  Relative atomic mass
# type_text   dtype = object Comment
#
#== Order of bonds of this type. numpy array.
#
# btype_order shape = (Nbtypes+1,) dtype = np.uint8
#
#== Link force field type.
#== numpy array. shape = (Nltype+1,) dtype = object. Present if chapter_present['Link Coeffs']
#
# btype_ff
# atype_ff
# dtype_ff
# itype_ff
#
#== Comments to link type.
#== numpy array. shape = (Nltype+1,) dtype = object. Present if chapter_present['Link Coeffs']
#
# btype_text
# atype_text
# dtype_text
# itype_text
#
#== Interaction parameters. numpy array, dtype = object. Points to list of str.
#== Present if chapter_present['... Coeffs']
#
# Pair_Coeffs              shape = (Ntypes+1,)
# PairIJ_Coeffs            shape = (Ntypes+1, Ntypes+1)
# Bond_Coeffs              shape = (Nbtypes+1,)
# Angle_Coeffs             shape = (Natypes+1,)
# BondBond Coeffs          shape = (Natypes+1,)
# BondAngle Coeffs         shape = (Natypes+1,)
# Dihedral_Coeffs          shape = (Ndtypes+1,)
# MiddleBondTorsion Coeffs shape = (Ndtypes+1,)
# EndBondTorsion Coeffs    shape = (Ndtypes+1,)
# AngleTorsion Coeffs      shape = (Ndtypes+1,)
# AngleAngleTorsion Coeffs shape = (Ndtypes+1,)
# BondBond13 Coeffs        shape = (Ndtypes+1,)
# Improper_Coeffs          shape = (Nitypes+1,)
# AngleAngle Coeffs        shape = (Nitypes+1,)

class c_cell:
    """Representation of atomistic system
    in a rectangular 3D periodic cell.
    """

    count_members = [
    'Ntypes', 'Nbtypes', 'Natypes', 'Ndtypes', 'Nitypes',
    'N', 'Nbonds', 'Nangles', 'Ndihedrals', 'Nimpropers',
    'Nmols'
    ]
    chapters = [
    'Masses', 'Pair Coeffs', 'PairIJ Coeffs',
    'Bond Coeffs', 'Angle Coeffs', 'Improper Coeffs', 'Dihedral Coeffs',
    'BondBond Coeffs', 'BondAngle Coeffs',
    'MiddleBondTorsion Coeffs', 'EndBondTorsion Coeffs',
    'AngleTorsion Coeffs', 'AngleAngleTorsion Coeffs', 'BondBond13 Coeffs',
    'AngleAngle Coeffs',
    'Atoms', 'Velocities',
    'Bonds', 'Angles', 'Dihedrals', 'Impropers'
    ]
    link_chapters = ['Bonds', 'Angles', 'Dihedrals', 'Impropers']
    coef_chapters = {
    'Pair Coeffs': 'Ntypes',
    'Bond Coeffs': 'Nbtypes',
    'Angle Coeffs': 'Natypes',
    'Dihedral Coeffs': 'Ndtypes',
    'Improper Coeffs': 'Nitypes',
    'BondBond Coeffs': 'Natypes',
    'BondAngle Coeffs': 'Natypes',
    'MiddleBondTorsion Coeffs': 'Ndtypes',
    'EndBondTorsion Coeffs': 'Ndtypes',
    'AngleTorsion Coeffs': 'Ndtypes',
    'AngleAngleTorsion Coeffs': 'Ndtypes',
    'BondBond13 Coeffs': 'Ndtypes',
    'AngleAngle Coeffs': 'Nitypes'
    }
    counts = [
    'atoms', 'bonds', 'angles', 'dihedrals', 'impropers',
    'atom types', 'bond types', 'angle types', 'dihedral types', 'improper types'
    ]

    def __init__(self, dataname = None):
        self.is_empty = True
        self.is_tilted = False
        self.comment = '\n'
        for cm in self.count_members:
            setattr(self, cm, 0)

#        self.type_m = np.zeros(1, dtype = float)
#        self.type_elm = np.empty(1, dtype='U2')
#        self.type_ff = np.empty(1, dtype='U8')
#        self.type_text = np.empty(1, dtype = object)
#
#        self.btype_order = np.zeros(1, dtype = np.uint8)
#        self.btype_ff = np.empty(1, dtype = object)
#        self.atype_ff = np.empty(1, dtype = object)
#        self.dtype_ff = np.empty(1, dtype = object)
#        self.itype_ff = np.empty(1, dtype = object)
#        self.btype_text = np.empty(1, dtype = object)
#        self.atype_text = np.empty(1, dtype = object)
#        self.dtype_text = np.empty(1, dtype = object)
#        self.itype_text = np.empty(1, dtype = object)
#
#        self.omol = np.zeros(1, dtype = int)
#        self.t = np.zeros(1, dtype = np.uint8)
#        self.q = np.zeros(1, dtype = float)
#        self.r = np.zeros((1, 3), dtype = float)
#        self.v = np.zeros((1, 3), dtype = float)
#        self.text = np.empty(1, dtype = object)
#
#        self.bond_t = np.zeros(1, dtype = np.uint8)
#        self.bond = np.zeros((1, 2), dtype = int)
#        self.bond_order = np.zeros(1, dtype = np.uint8)
#
#        self.angle_t = np.zeros(1, dtype = np.uint8)
#        self.angle = np.zeros((1, 3), dtype = int)
#
#        self.dihedral_t = np.zeros(1, dtype = np.uint8)
#        self.dihedral = np.zeros((1, 4), dtype = int)
#
#        self.improper_t = np.zeros(1, dtype = np.uint8)
#        self.improper = np.zeros((1, 4), dtype = int)
#
#        self.mol_name = np.empty(1, dtype = object)
#        self.mol_1st = np.zeros(1, dtype = int)
#        self.mol_b1st = np.zeros(1, dtype = int)
#        self.mol_a1st = np.zeros(1, dtype = int)
#        self.mol_d1st = np.zeros(1, dtype = int)
#        self.mol_i1st = np.zeros(1, dtype = int)

        self.xlo = 0.0; self.xhi = 0.0
        self.ylo = 0.0; self.yhi = 0.0
        self.zlo = 0.0; self.zhi = 0.0

        self.chapter_present = {ch: False for ch in self.chapters}
        self.chapter_comment = {ch: '' for ch in self.chapters}

        if dataname != None:
            self.parse_data(dataname)


    def size(self):
        return np.array([self.xhi-self.xlo, self.yhi-self.ylo, self.zhi-self.zlo])


    def adjust_size(self, margin = 1.0):
        "Adjust cell boundaries to enclose all atoms"

        if self.is_empty:
            return

        xmin = np.min(self.r[:, 0])
        xmax = np.max(self.r[:, 0])
        ymin = np.min(self.r[:, 1])
        ymax = np.max(self.r[:, 1])
        zmin = np.min(self.r[:, 2])
        zmax = np.max(self.r[:, 2])

        self.xlo = xmin - margin; self.xhi = xmax + margin
        self.ylo = ymin - margin; self.yhi = ymax + margin
        self.zlo = zmin - margin; self.zhi = zmax + margin


    def masses(self):
        return self.type_m[self.t]

    def density(self):
        "Returns density of the system in g/cm3"
        xsize, ysize, zsize = self.size()
        V = xsize * ysize * zsize * 1e-24
        M = np.sum(self.type_m[self.t[1:]]) / 6.022141e23
        return M / V

    def com(self):
        m = self.type_m[self.t]
        m[0] = 0.0
        M = np.sum(m)
        ret = np.sum(self.r * m.reshape(self.N+1, 1) / M, axis = 0)
        return ret


    def orient(self):
        m = self.type_m[self.t[1:]]
        self.r[1:], xaxis, yaxis, zaxis = orient(self.r[1:], m)
        if self.chapter_present['Velocities']:
            aux = np.empty((self.N, 3))
            aux[:, 0] = np.inner(self.v[1:], xaxis)
            aux[:, 1] = np.inner(self.v[1:], yaxis)
            aux[:, 2] = np.inner(self.v[1:], zaxis)
            self.v[1:] = aux


    def in_cell(self):
        clo = np.array([self.xlo, self.ylo, self.zlo])
        size = self.size()
        m = self.type_m[self.t]
        for omol in range(1, self.Nmols+1):
            beg = self.mol_1st[omol]
            end = self.mol_1st[omol+1]
            N = end - beg
            M = np.sum(m[beg:end])
            com = np.sum(self.r[beg:end] * m[beg:end].reshape((N, 1)), axis = 0) / M
            new_com = (com - clo) % size + clo
            self.r[beg:end] += new_com - com


    def arrange_filters(self, mol_options, atom_options, union = False):
        """mol_options and atom_options must be dictionaries, even empty"""

        AtomFilter = set()
        MolFilter = set()

        def read_filter_file(filter_file_name, accept_strings = False):
            subsetfile = open(filter_file_name, 'r')
            ret = set()
            for line in subsetfile:
                for token in line.split():
                    try:
                        ret.add(int(token))
                    except ValueError:
                        if accept_strings:
                            ret.add(token)
            subsetfile.close()
            return ret

        if len(mol_options) > 0:
            if not union:
                MolFilter = set(range(1, self.Nmols+1))
            if 'mol_filter' in mol_options:
                mol_filter = mol_options['mol_filter']
                MolFilter = read_filter_file(mol_filter, accept_strings = True)
                MolFilter = {o for o in range(1, self.Nmols+1) if o in MolFilter or self.mol_name[o] in MolFilter}
            if 'mol_filter_out' in mol_options:
                mol_filter_out = mol_options['mol_filter_out']
                MolFilterOut = read_filter_file(mol_filter_out, accept_strings = True)
                if union:
                    MolFilterPart = {o for o in range(1, self.Nmols+1) if o not in MolFilterOut and self.mol_name[o] not in MolFilterOut}
                    MolFilter = MolFilter.union(MolFilterPart)
                else:
                    MolFilter = {o for o in MolFilter if o not in MolFilterOut and self.mol_name[o] not in MolFilterOut}
            if 'mol_list' in mol_options:
                mol_list = mol_options['mol_list']
                mol_list = set(mol_list)
                if union:
                    MolFilterPart = {o for o in range(1, self.Nmols+1) if o in mol_list or self.mol_name[o] in mol_list}
                    MolFilter = MolFilter.union(MolFilterPart)
                else:
                    MolFilter = {o for o in MolFilter if o in mol_list or self.mol_name[o] in mol_list}
            if 'mol_list_out' in mol_options:
                mol_list_out = mol_options['mol_list_out']
                mol_list_out = set(mol_list_out)
                if union:
                    MolFilterPart = {o for o in range(1, self.Nmols+1) if o not in mol_list_out and self.mol_name[o] not in mol_list_out}
                    MolFilter = MolFilter.union(MolFilterPart)
                else:
                    MolFilter = {o for o in MolFilter if o not in mol_list_out and self.mol_name[o] not in mol_list_out}
            if 'mol_range' in mol_options:
                mol_range = mol_options['mol_range']
                olo = min(mol_range)
                ohi = max(mol_range)
                olo = min(olo, self.Nmols); ohi = min(ohi, self.Nmols)
                olo = max(olo, 1); ohi = max(ohi, 1)
                MolFilterPart = set(range(olo, ohi+1))
                if union:
                    MolFilter = MolFilter.union(MolFilterPart)
                else:
                    MolFilter = MolFilter.intersection(MolFilterPart)
            if 'mol_range_out' in mol_options:
                mol_range_out = mol_options['mol_range_out']
                olo = min(mol_range_out)
                ohi = max(mol_range_out)
                olo = min(olo, self.Nmols); ohi = min(ohi, self.Nmols)
                olo = max(olo, 1); ohi = max(ohi, 1)
                if union:
                    MolFilter = MolFilter.union([set(range(1, olo)), set(range(ohi+1, self.Nmols+1))])
                else:
                    MolFilter = MolFilter.difference(range(olo, ohi+1))
            if 'mol_re' in mol_options:
                mol_re = mol_options['mol_re']
                reo = re.compile(mol_re)
                if union:
                    MolFilterPart = {o for o in range(1, self.Nmols+1) if (self.mol_name[o] and reo.fullmatch(self.mol_name[o]))}
                    MolFilter = MolFilter.union(MolFilterPart)
                else:
                    MolFilter = {o for o in MolFilter if (self.mol_name[o] and reo.fullmatch(self.mol_name[o]))}
            if 'mol_re_out' in mol_options:
                mol_re_out = mol_options['mol_re_out']
                reo = re.compile(mol_re_out)
                if union:
                    MolFilterPart = {o for o in range(1, self.Nmols+1) if (not self.mol_name[o] or not reo.fullmatch(self.mol_name[o]))}
                    MolFilter = MolFilter.union(MolFilterPart)
                else:
                    MolFilter = {o for o in MolFilter if (not self.mol_name[o] or not reo.fullmatch(self.mol_name[o]))}

        if len(atom_options) > 0:
            if len(MolFilter) > 0:
                for omol in MolFilter:
                    AtomFilter = AtomFilter.union(set(range(self.mol_1st[omol], self.mol_1st[omol+1])))
                MolFilter = set()
            elif not union:
                AtomFilter = set(range(1, self.N+1))
            if 'atom_filter' in atom_options:
                atom_filter = atom_options['atom_filter']
                atom_filter = read_filter_file(atom_filter)
                if union:
                    AtomFilterPart = {o for o in atom_filter if 1 <= o <= self.N}
                    AtomFilter = AtomFilter.union(AtomFilterPart)
                else:
                    AtomFilter = AtomFilter.intersection(atom_filter)
            if 'atom_filter_out' in atom_options:
                atom_filter_out = atom_options['atom_filter_out']
                atom_filter_out = read_filter_file(atom_filter)
                if union:
                    AtomFilterPart = {o for o in range(1, self.N+1) if o not in atom_filter_out}
                    AtomFilter = AtomFilter.union(AtomFilterPart)
                else:
                    AtomFilter = AtomFilter.difference(atom_filter_out)
            if 'atom_list' in atom_options:
                atom_list = atom_options['atom_list']
                atom_list = set(atom_list)
                if union:
                    AtomFilter = AtomFilter.union(atom_list)
                else:
                    AtomFilter = AtomFilter.intersection(atom_list)
            if 'atom_list_out' in atom_options:
                atom_list_out = atom_options['atom_list_out']
                atom_list_out = set(atom_list_out)
                if union:
                    AtomFilterPart = {o for o in range(1, self.N+1) if o not in atom_list_out}
                    AtomFilter = AtomFilter.union(AtomFilterPart)
                else:
                    AtomFilter = AtomFilter.difference(atom_list_out)
            if 'atom_range' in atom_options:
                atom_range = atom_options['atom_range']
                olo = min(atom_range)
                ohi = max(atom_range)
                olo = min(olo, self.N); ohi = min(ohi, self.N)
                olo = max(olo, 1); ohi = max(ohi, 1)
                AtomFilterPart = set(range(olo, ohi+1))
                if union:
                    AtomFilter = AtomFilter.union(AtomFilterPart)
                else:
                    AtomFilter = AtomFilter.intersection(AtomFilterPart)
            if 'atom_range_out' in atom_options:
                atom_range_out = atom_options['atom_range_out']
                olo = min(atom_range_out)
                ohi = max(atom_range_out)
                olo = min(olo, self.N); ohi = min(ohi, self.N)
                olo = max(olo, 1); ohi = max(ohi, 1)
                if union:
                    AtomFilter = AtomFilter.union([set(range(1, olo)), set(range(ohi+1, self.Nmols+1))])
                else:
                    AtomFilter = AtomFilter.difference(range(olo, ohi+1))
            if 'atom_types' in atom_options:
                atom_types = atom_options['atom_types']
                atom_types = set(atom_types)
                if union:
                    AtomFilterPart = {o for o in range(1, self.N+1) if self.t[o] in atom_types}
                    AtomFilter = AtomFilter.union(AtomFilterPart)
                else:
                    AtomFilter = {o for o in AtomFilter if self.t[o] in atom_types}
            if 'atom_types_out' in atom_options:
                atom_types_out = atom_options['atom_types_out']
                atom_types_out = set(atom_types_out)
                if union:
                    AtomFilterPart = {o for o in range(1, self.N+1) if self.t[o] not in atom_types_out}
                    AtomFilter = AtomFilter.union(AtomFilterPart)
                else:
                    AtomFilter = {o for o in AtomFilter if self.t[o] not in atom_types_out}
            if 'atom_fftypes' in atom_options:
                atom_fftypes = atom_options['atom_fftypes']
                atom_fftypes = set(atom_fftypes)
                ff = self.type_ff[self.t]
                if union:
                    AtomFilterPart = {o for o in range(1, self.N+1) if ff[o] in atom_fftypes}
                    AtomFilter = AtomFilter.union(AtomFilterPart)
                else:
                    AtomFilter = {o for o in AtomFilter if ff[o] in atom_fftypes}
            if 'atom_fftypes_out' in atom_options:
                atom_fftypes_out = atom_options['atom_fftypes_out']
                atom_fftypes_out = set(atom_fftypes_out)
                ff = self.type_ff[self.t]
                if union:
                    AtomFilterPart = {o for o in range(1, self.N+1) if ff[o] not in atom_fftypes_out}
                    AtomFilter = AtomFilter.union(AtomFilterPart)
                else:
                    AtomFilter = {o for o in AtomFilter if ff[o] not in atom_fftypes_out}
            if 'atom_text_re' in atom_options:
                atom_text_re = atom_options['atom_text_re']
                reo = re.compile(atom_text_re)
                if union:
                    AtomFilterPart = {o for o in range(1, self.N+1) if (self.text[o] and reo.fullmatch(self.text[o]))}
                    AtomFilter = AtomFilter.union(AtomFilterPart)
                else:
                    AtomFilter = {o for o in AtomFilter if (self.text[o] and reo.fullmatch(self.text[o]))}
            if 'atom_text_re_out' in atom_options:
                atom_text_re_out = atom_options['atom_text_re_out']
                reo = re.compile(atom_text_re_out)
                if union:
                    AtomFilterPart = {o for o in range(1, self.N+1) if (not self.text[o] or not reo.fullmatch(self.text[o]))}
                    AtomFilter = AtomFilter.union(AtomFilterPart)
                else:
                    AtomFilter = {o for o in AtomFilter if (not self.text[o] or not reo.fullmatch(self.text[o]))}

        if len(MolFilter) == 0:
            MolFilter = None
        if len(AtomFilter) == 0:
            AtomFilter = None

        return MolFilter, AtomFilter


    def vdw_radii(self, forcefield):
        ff_module = import_module(forcefield)
        ff = ff_module.c_ff({})
        type_radii = np.zeros(self.Ntypes+1)       # half Lennard-Jones' sigma for each force field type
        mult = (2**(1./6.))/2
        for ot in range(1, self.Ntypes+1):
            type_radii[ot] = ff.ff_pair[self.type_ff[ot]][-1] * mult
        radii = np.zeros(self.N+1)
        radii[1:] = type_radii[self.t[1:]]
        return radii


    def parse_zmatrix(self, zname,
    prev = np.array([[0., 1., 0.], [1., 1., 0.], [1., 0., 0.]]),
    l1 = 1., a1 = 90, d1 = 0, a2 = 90, d2 = 0, d3 = 0,
    constants = {}):
        """
        Parse a kind of z-matrix file (.zdat)
        constants dictionary has a priority over the values
        defined in the Constants chapter of .zdat file
        """

        data = open(zname, 'r')

        desc = []
        toc = [[]]
        hdr = True
        for line in data:
            line = line[:-1]
            l = line.split('#')
            mainline = l[0]
            values = mainline.split()
            comment = '#'.join(l[1:])
            comment = comment.lstrip()
            if hdr:
                if len(values) == 0:
                    if comment != '':
                        if self.comment == '\n':
                            self.comment = comment + '\n'
                        else:
                            self.comment += comment + '\n'
                    continue
                iscount = False
                for ct in self.counts:
                    if ct in mainline:
                        N = int(values[0])
                        if ct == 'atoms':
                            self.N = N
                            self.omol = np.zeros(N+1, dtype = int)
                            self.t = np.zeros(N+1, dtype = np.uint8)
                            self.q = np.zeros(N+1, dtype = float)
                            self.r = np.zeros((N+1, 3), dtype = float)
                            self.v = np.zeros((N+1, 3), dtype = float)
                            self.text = np.empty(N+1, dtype = object)
#                        elif ct == 'bonds':
#                            self.Nbonds = N
#                            self.bond_t = np.zeros(N+1, dtype = np.uint8)
#                            self.bond = np.zeros((N+1, 2), dtype = int)
#                            self.bond_order = np.zeros(N+1, dtype = np.uint8)
                        elif ct == 'angles':
                            self.Nangles = N
                            self.angle_t = np.zeros(N+1, dtype = np.uint8)
                            self.angle = np.zeros((N+1, 3), dtype = int)
                        elif ct == 'dihedrals':
                            self.Ndihedrals = N
                            self.dihedral_t = np.zeros(N+1, dtype = np.uint8)
                            self.dihedral = np.zeros((N+1, 4), dtype = int)
                        elif ct == 'impropers':
                            self.Nimpropers = N
                            self.improper_t = np.zeros(N+1, dtype = np.uint8)
                            self.improper = np.zeros((N+1, 4), dtype = int)
                        elif ct == 'atom types':
                            self.Ntypes = N
                            self.type_m = np.zeros(N+1, dtype = float)
                            self.type_elm = np.empty(N+1, dtype='U2')
                            self.type_ff = np.empty(N+1, dtype='U8')
                            self.type_text = np.empty(N+1, dtype = object)
                        elif ct == 'bond types':
                            self.Nbtypes = N
                            self.btype_order = np.zeros(N+1, dtype = np.uint8)
                            self.btype_ff = np.empty(N+1, dtype = object)
                            self.btype_text = np.empty(N+1, dtype = object)
                        elif ct == 'angle types':
                            self.Natypes = N
                            self.atype_ff = np.empty(N+1, dtype = object)
                            self.atype_text = np.empty(N+1, dtype = object)
                        elif ct == 'dihedral types':
                            self.Ndtypes = N
                            self.dtype_ff = np.empty(N+1, dtype = object)
                            self.dtype_text = np.empty(N+1, dtype = object)
                        elif ct == 'improper types':
                            self.Nitypes = N
                            self.itype_ff = np.empty(N+1, dtype = object)
                            self.itype_text = np.empty(N+1, dtype = object)
                        iscount = True
                        break
                if iscount:
                    continue
                if 'xlo' in mainline:
                    self.xlo = float(values[0])
                    self.xhi = float(values[1])
                    continue
                if 'ylo' in mainline:
                    self.ylo = float(values[0])
                    self.yhi = float(values[1])
                    continue
                if 'zlo' in mainline:
                    self.zlo = float(values[0])
                    self.zhi = float(values[1])
                    continue
                if 'xy' in mainline:
                    self.xy = float(values[0])
                    self.xz = float(values[1])
                    self.yz = float(values[2])
                    self.is_tilted = True
                    continue
                hdr = False
            if len(values) == 0: continue
            v0 = values[0]
            if not (v0.isdigit() or (v0[0] == '-' and v0[1:].isdigit())):
                ch = ' '.join(values)
                if ch in self.chapters or ch == 'Constants':
                    toc[-1].append(len(desc))
                    toc.append([ch, len(desc) + 1])
                    self.chapter_present[ch] = True
                    self.chapter_comment[ch] = comment
                else:
                    raise Exception('Unknown keyword: ' + ch)
            desc.append((values, comment))
        toc[-1].append(len(desc))
        toc.pop(0)

        data.close()

        bonds = [(0, 0, 0, 0)]    # (oatom0, oatom1, type, order)
        acceptable_orders = {'1': 1, '2': 2, '3': 3, '6': 6, '7': 7, 'a': 6, '1.5': 7}

        constants['e'] = np.e
        constants['pi'] = np.pi
        constants['sin'] = np.sin
        constants['cos'] = np.cos
        constants['tan'] = np.tan
        constants['arcsin'] = np.arcsin
        constants['arccos'] = np.arccos
        constants['arctan'] = np.arctan
        constants['sinh'] = np.sinh
        constants['cosh'] = np.cosh
        constants['tanh'] = np.tanh
        constants['arcsinh'] = np.arcsinh
        constants['arccosh'] = np.arccosh
        constants['arctanh'] = np.arctanh
        constants['exp'] = np.exp
        constants['log'] = np.log
        constants['log10'] = np.log10
        constants['sign'] = np.sign

        for chp_name, chp_start, chp_end in toc:
            if chp_name == 'Constants':
                for v, comment in desc[chp_start:chp_end]:
                    for i in range(1, len(v)):
                        if '=' in v[i]:
                            constname, aftereq = v[i].split('=')
                            if constname == '':
                                constname = v[i-1]
                            if constname not in constants:
                                expression = ' '.join([aftereq] + v[i+1:])
                                c = eval(expression, constants)
                                constants[constname] = c
                            break
                break

        ffdict = {}

        for chp_name, chp_start, chp_end in toc:

            if chp_name == 'Masses':
                for v, comment in desc[chp_start:chp_end]:
                    o = int(v[0])
                    self.type_m[o] = m = float(v[1])
                    c = comment.split()
                    if len(c) > 0:
                        if len(c[0]) <=2:
                            self.type_elm[o] = c[0]
                        else:
                            self.type_text[o] = ' '.join(c)
                            c = []
                    if len(c) == 0:
                        is_real = False
                        for e in ptable.elements:
                            if abs(m - e.m) < 0.5:
                                self.type_elm[o] = e.symbol
                                is_real = True
                                break
                        if not is_real:
                            self.type_elm[o] = 'XX'
                    if len(c) >= 2:
                        self.type_ff[o] = c[1]
                    else:
                        self.type_ff[o] = self.type_elm[o] + v[0]
                    if len(c) >= 3:
                        self.type_text[o] = ' '.join(c[2:])
                for ot in range(1, self.Ntypes+1):
                    ffdict[self.type_ff[ot]] = ot
                ffdict['X'] = 0

            elif chp_name == 'PairIJ Coeffs':
                self.PairIJ_Coeffs = np.empty((self.Ntypes+1, self.Ntypes+1), dtype = object)
                for v, comment in desc[chp_start:chp_end]:
                    o0 = int(v[0])
                    o1 = int(v[1])
                    self.PairIJ_Coeffs[o0, o1] = v[2:]

            elif chp_name in self.coef_chapters:
                attr_name = chp_name.replace(' ', '_')
                N = getattr(self, c_cell.coef_chapters[chp_name])
                coeffs = np.empty(N+1, dtype = object)
                setattr(self, attr_name, coeffs)
                for v, comment in desc[chp_start:chp_end]:
                    o = int(v[0])
                    coeffs[o] = v[1:]
                    if comment != '':
                        c = comment.split()
                        if chp_name == 'Bond Coeffs':
                            if c[0] in acceptable_orders:
                                self.btype_order[o] = acceptable_orders[c.pop(0)]
                            self.btype_ff[o] = c.pop(0) if len(c) > 0 else None
                            self.btype_text[o] = ' '.join(c) if len(c) > 0 else None
                        elif chp_name == 'Angle Coeffs':
                            self.atype_ff[o] = c.pop(0)
                            self.atype_text[o] = ' '.join(c) if len(c) > 0 else None
                        elif chp_name == 'Dihedral Coeffs':
                            self.dtype_ff[o] = c.pop(0)
                            self.dtype_text[o] = ' '.join(c) if len(c) > 0 else None
                        elif chp_name == 'Improper Coeffs':
                            self.itype_ff[o] = c.pop(0)
                            self.itype_text[o] = ' '.join(c) if len(c) > 0 else None

            elif chp_name == 'Atoms':
                mol_1st = [0]
                mol_name = [None]
                xatoms = {0: np.zeros(3)}
                curomol = 0
                orecord = 0
                for v, comment in desc[chp_start:chp_end]:
                    bond_type = 0
                    bond_order = 1

                    # o omol type bond_type q o0 l o1 a o2 d # mol_name bond_order comment
                    orecord += 1
                    if orecord == 1:
                        r0 = prev[0]
                        l = l1
                        r1 = prev[1]
                        a = a1
                        r2 = prev[2]
                        d = d1
                        o0 = 0
                    else:
                        o0 = int(v[5])
                        if o0 > 0:
                            r0 = self.r[o0]
                        else:
                            r0 = xatoms[o0]
                        try:
                            l = float(v[6])
                        except ValueError:
                            l = eval(v[6], constants)
                    if orecord == 2:
                        r1 = prev[0]
                        a = a2
                        r2 = prev[1]
                        d = d2
                    elif orecord > 2:
                        o1 = int(v[7])
                        if o1 > 0:
                            r1 = self.r[o1]
                        else:
                            r1 = xatoms[o1]
                        try:
                            a = float(v[8])
                        except ValueError:
                            a = eval(v[8], constants)
                    if orecord == 3:
                        r2 = prev[0]
                        d = d3
                    elif orecord > 3:
                        o2 = int(v[9])
                        if o2 > 0:
                            r2 = self.r[o2]
                        else:
                            r2 = xatoms[o2]
                        try:
                            d = float(v[10])
                        except ValueError:
                            d = eval(v[10], constants)
                    r = zpoint(r0, l, r1, a, r2, d)

                    o = int(v[0])
                    if o > 0:
                        self.omol[o] = omol = int(v[1])
                        if v[2].isdigit():
                            self.t[o] = int(v[2])
                        else:
                            self.t[o] = ffdict[v[2]]
                        if o0 > 0:
                            bond_type = int(v[3])
                        try:
                            self.q[o] = float(v[4])
                        except ValueError:
                            self.q[o] = eval(v[4], constants)
                        self.r[o, :] = r
                    else:
                        if o in xatoms:
                            raise Exception("Repeating fantom atom number " + str(o))
                        xatoms[o] = r

                    c = comment.split()
                    if o > 0 and omol != curomol:
                        bond_type = 0
                        mol_1st.append(o)
                        if len(c) > 0:
                            mol_name.append(c[0])
                        else:
                            mol_name.append('XXXX_' + str(omol))
                        curomol = omol
                    if o > 0 and len(c) > 1:
                        if c[1] in acceptable_orders:
                            bond_order = acceptable_orders[c[1]]
                            if len(c) > 2:
                                self.text[o] = ' '.join(c[2:])
                        else:
                            self.text[o] = ' '.join(c[1:])

                    if bond_type != 0:
                        bonds.append( (o0, o, bond_type, bond_order) )

                self.Nmols = len(mol_name) - 1
                mol_1st.append(self.N + 1)
                self.mol_1st = np.array(mol_1st, dtype = int)
                self.mol_name = np.array(mol_name, dtype = object)

            elif chp_name == 'Velocities':
                for v, comment in desc[chp_start:chp_end]:
                    o = int(v[0])
                    self.v[o] = [float(vvv) for vvv in v[1:4]]

            elif chp_name == 'Bonds':
                for v, comment in desc[chp_start:chp_end]:
                    iv = [int(vvv) for vvv in v]
                    bond_order = 1
                    c = comment.split()
                    if len(c) > 0:
                        if c[0] in acceptable_orders:
                            bond_order = acceptable_orders[c[0]]
                    bonds.append( (iv[2], iv[3], iv[1], bond_order) )

            elif chp_name in self.link_chapters:
                link = getattr(self, chp_name[:-1].lower())
                link_n = link.shape[-1]
                link_t = getattr(self, chp_name[:-1].lower() + '_t')
                Nlinks = getattr(self, 'N' + chp_name.lower())
                mol_l1st = np.zeros(self.Nmols + 2, dtype = int)
                curomol = 0
                for v, comment in desc[chp_start:chp_end]:
                    iv = [int(vvv) for vvv in v]
                    o = iv[0]
                    link_t[o] = iv[1]
                    link[o] = iv[2: (2 + link_n)]
                    omol = self.omol[link[o, 0]]
                    if omol != curomol:
                        for om in range(curomol+1, omol+1):
                            mol_l1st[om] = o
                        curomol = omol
                mol_l1st[curomol+1:] = Nlinks + 1
                setattr(self, 'mol_' + chp_name[0].lower() + '1st', mol_l1st)

        bonds.sort()
        bonds = np.array(bonds, dtype = int)
        self.Nbonds = bonds.shape[0] - 1
        if self.Nbonds > 0:
            self.chapter_present['Bonds'] = True
        self.bond = np.zeros((self.Nbonds+1, 2), dtype = int)
        self.bond_t = np.zeros(self.Nbonds+1, dtype = int)
        self.bond_order = np.zeros(self.Nbonds+1, dtype = int)
        self.bond[:, :] = bonds[:, :2]
        self.bond_t[:] = bonds[:, 2]
        self.bond_order[:] = bonds[:, 3]
        self.mol_b1st = np.zeros(self.Nmols + 2, dtype = int)
        curomol = 0
        for o in range(1, self.Nbonds+1):
            omol = self.omol[bonds[o, 0]]
            if omol != curomol:
                for om in range(curomol+1, omol+1):
                    self.mol_b1st[om] = o
                curomol = omol
        self.mol_b1st[curomol+1:] = self.Nbonds + 1

        self.is_empty = False


    def parse_data(self, dataname):
        'Parses LAMMPS data file into c_cell'

        data = open(dataname, 'r')

        desc = []
        toc = [[]]
        hdr = True
        for line in data:
            line = line[:-1]
            l = line.split('#')
            mainline = l[0]
            values = mainline.split()
            comment = '#'.join(l[1:])
            comment = comment.lstrip()
            if hdr:
                if len(values) == 0:
                    if comment != '':
                        if self.comment == '\n':
                            self.comment = comment + '\n'
                        else:
                            self.comment += comment + '\n'
                    continue
                iscount = False
                for ct in self.counts:
                    if ct in mainline:
                        N = int(values[0])
                        if ct == 'atoms':
                            self.N = N
                            self.omol = np.zeros(N+1, dtype = int)
                            self.t = np.zeros(N+1, dtype = np.uint8)
                            self.q = np.zeros(N+1, dtype = float)
                            self.r = np.zeros((N+1, 3), dtype = float)
                            self.v = np.zeros((N+1, 3), dtype = float)
                            self.text = np.empty(N+1, dtype = object)
                        elif ct == 'bonds':
                            self.Nbonds = N
                            self.bond_t = np.zeros(N+1, dtype = np.uint8)
                            self.bond = np.zeros((N+1, 2), dtype = int)
                            self.bond_order = np.zeros(N+1, dtype = np.uint8)
                        elif ct == 'angles':
                            self.Nangles = N
                            self.angle_t = np.zeros(N+1, dtype = np.uint8)
                            self.angle = np.zeros((N+1, 3), dtype = int)
                        elif ct == 'dihedrals':
                            self.Ndihedrals = N
                            self.dihedral_t = np.zeros(N+1, dtype = np.uint8)
                            self.dihedral = np.zeros((N+1, 4), dtype = int)
                        elif ct == 'impropers':
                            self.Nimpropers = N
                            self.improper_t = np.zeros(N+1, dtype = np.uint8)
                            self.improper = np.zeros((N+1, 4), dtype = int)
                        elif ct == 'atom types':
                            self.Ntypes = N
                            self.type_m = np.zeros(N+1, dtype = float)
                            self.type_elm = np.empty(N+1, dtype='U2')
                            self.type_ff = np.empty(N+1, dtype='U8')
                            self.type_text = np.empty(N+1, dtype = object)
                        elif ct == 'bond types':
                            self.Nbtypes = N
                            self.btype_order = np.zeros(N+1, dtype = np.uint8)
                            self.btype_ff = np.empty(N+1, dtype = object)
                            self.btype_text = np.empty(N+1, dtype = object)
                        elif ct == 'angle types':
                            self.Natypes = N
                            self.atype_ff = np.empty(N+1, dtype = object)
                            self.atype_text = np.empty(N+1, dtype = object)
                        elif ct == 'dihedral types':
                            self.Ndtypes = N
                            self.dtype_ff = np.empty(N+1, dtype = object)
                            self.dtype_text = np.empty(N+1, dtype = object)
                        elif ct == 'improper types':
                            self.Nitypes = N
                            self.itype_ff = np.empty(N+1, dtype = object)
                            self.itype_text = np.empty(N+1, dtype = object)
                        iscount = True
                        break
                if iscount:
                    continue
                if 'xlo' in mainline:
                    self.xlo = float(values[0])
                    self.xhi = float(values[1])
                    continue
                if 'ylo' in mainline:
                    self.ylo = float(values[0])
                    self.yhi = float(values[1])
                    continue
                if 'zlo' in mainline:
                    self.zlo = float(values[0])
                    self.zhi = float(values[1])
                    continue
                if 'xy' in mainline:
                    self.xy = float(values[0])
                    self.xz = float(values[1])
                    self.yz = float(values[2])
                    self.is_tilted = True
                    continue
                hdr = False
            if len(values) == 0: continue
            if not values[0].isdigit():
                ch = ' '.join(values)
                if ch in self.chapters:
                    toc[-1].append(len(desc))
                    toc.append([ch, len(desc) + 1])
                    self.chapter_present[ch] = True
                    self.chapter_comment[ch] = comment
                else:
                    raise Exception('Unknown keyword: ' + ch)
            desc.append((values, comment))
        toc[-1].append(len(desc))
        toc.pop(0)

        data.close()

        acceptable_orders = {'1': 1, '2': 2, '3': 3, '6': 6, '7': 7, 'a': 6, '1.5': 7}

        for chp_name, chp_start, chp_end in toc:

            if chp_name == 'Masses':
                for v, comment in desc[chp_start:chp_end]:
                    o = int(v[0])
                    self.type_m[o] = m = float(v[1])
                    c = comment.split()
                    if len(c) > 0:
                        if len(c[0]) <=2:
                            self.type_elm[o] = c[0]
                        else:
                            self.type_text[o] = ' '.join(c)
                            c = []
                    if len(c) == 0:
                        is_real = False
                        for e in ptable.elements:
                            if abs(m - e.m) < 0.5:
                                self.type_elm[o] = e.symbol
                                is_real = True
                                break
                        if not is_real:
                            self.type_elm[o] = 'XX'
                    if len(c) >= 2:
                        self.type_ff[o] = c[1]
                    else:
                        self.type_ff[o] = self.type_elm[o] + v[0]
                    if len(c) >= 3:
                        self.type_text[o] = ' '.join(c[2:])

            elif chp_name == 'PairIJ Coeffs':
                self.PairIJ_Coeffs = np.empty((self.Ntypes+1, self.Ntypes+1), dtype = object)
                for v, comment in desc[chp_start:chp_end]:
                    o0 = int(v[0])
                    o1 = int(v[1])
                    self.PairIJ_Coeffs[o0, o1] = v[2:]

            elif chp_name in self.coef_chapters:
                attr_name = chp_name.replace(' ', '_')
                N = getattr(self, c_cell.coef_chapters[chp_name])
                coeffs = np.empty(N+1, dtype = object)
                setattr(self, attr_name, coeffs)
                for v, comment in desc[chp_start:chp_end]:
                    o = int(v[0])
                    coeffs[o] = v[1:]
                    if comment != '':
                        c = comment.split()
                        if chp_name == 'Bond Coeffs':
                            if c[0] in acceptable_orders:
                                self.btype_order[o] = acceptable_orders[c.pop(0)]
                            self.btype_ff[o] = c.pop(0) if len(c) > 0 else None
                            self.btype_text[o] = ' '.join(c) if len(c) > 0 else None
                        elif chp_name == 'Angle Coeffs':
                            self.atype_ff[o] = c.pop(0)
                            self.atype_text[o] = ' '.join(c) if len(c) > 0 else None
                        elif chp_name == 'Dihedral Coeffs':
                            self.dtype_ff[o] = c.pop(0)
                            self.dtype_text[o] = ' '.join(c) if len(c) > 0 else None
                        elif chp_name == 'Improper Coeffs':
                            self.itype_ff[o] = c.pop(0)
                            self.itype_text[o] = ' '.join(c) if len(c) > 0 else None

            elif chp_name == 'Atoms':
                mol_1st = [0]
                mol_name = [None]
                curomol = 0
                for v, comment in desc[chp_start:chp_end]:
                    o = int(v[0])
                    self.omol[o] = omol = int(v[1])
                    self.t[o] = int(v[2])
                    self.q[o] = float(v[3])
                    self.r[o] = [float(vvv) for vvv in v[4:7]]

                    c = comment.split()
                    if omol != curomol:
                        mol_1st.append(o)
                        if len(c) > 0:
                            mol_name.append(c[0])
                        else:
                            mol_name.append('XXXX_' + str(omol))
                        curomol = omol
                    if len(c) > 1:
                        self.text[o] = ' '.join(c[1:])
                    else:
                        self.text[o] = ''
                self.Nmols = len(mol_name) - 1
                mol_1st.append(self.N + 1)
                self.mol_1st = np.array(mol_1st, dtype = int)
                self.mol_name = np.array(mol_name, dtype = object)

            elif chp_name == 'Velocities':
                for v, comment in desc[chp_start:chp_end]:
                    o = int(v[0])
                    self.v[o] = [float(vvv) for vvv in v[1:4]]

            elif chp_name in self.link_chapters:
                is_bond = (chp_name == 'Bonds')
                link = getattr(self, chp_name[:-1].lower())
                link_n = link.shape[-1]
                link_t = getattr(self, chp_name[:-1].lower() + '_t')
                Nlinks = getattr(self, 'N' + chp_name.lower())
                mol_l1st = np.zeros(self.Nmols + 2, dtype = int)
                curomol = 0
                for v, comment in desc[chp_start:chp_end]:
                    iv = [int(vvv) for vvv in v]
                    o = iv[0]
                    link_t[o] = iv[1]
                    link[o] = iv[2: (2 + link_n)]
                    if is_bond:
                        c = comment.split()
                        if len(c) > 0:
                            if c[0] in acceptable_orders:
                                self.bond_order[o] = acceptable_orders[c[0]]
                    omol = self.omol[link[o, 0]]
                    if omol != curomol:
                        for om in range(curomol+1, omol+1):
                            mol_l1st[om] = o
                        curomol = omol
                mol_l1st[curomol+1:] = Nlinks + 1
                setattr(self, 'mol_' + chp_name[0].lower() + '1st', mol_l1st)

        self.is_empty = False

    def write_data(self, newdataname, molfilter = None, atomfilter = None):
        'Converts cell structure back into LAMMPS data file'

        if self.is_empty:
            return

        is_filtered = (molfilter != None) or (atomfilter != None)
        if is_filtered:
            new_o = np.zeros(self.N+1, dtype = int)
            new_ob = np.zeros(self.Nbonds+1, dtype = int)
            new_oa = np.zeros(self.Nangles+1, dtype = int)
            new_od = np.zeros(self.Ndihedrals+1, dtype = int)
            new_oi = np.zeros(self.Nimpropers+1, dtype = int)
            new_om = np.zeros(self.Nmols+1, dtype = int)
            new_ol_dict = {'Bonds': new_ob, 'Angles': new_oa,
            'Dihedrals': new_od, 'Impropers': new_oi}

        N = self.N
        Nbonds = self.Nbonds
        Nangles = self.Nangles
        Ndihedrals = self.Ndihedrals
        Nimpropers = self.Nimpropers
        chapter_present = copy.deepcopy(self.chapter_present)

        if atomfilter != None:
            atomfilter = set(atomfilter)
            Nbonds = 0; Nangles = 0
            Ndihedrals = 0; Nimpropers = 0
            cur_o = 0
            for o in range(1, self.N+1):
                if o in atomfilter:
                    cur_o += 1
                    new_o[o] = cur_o
            N = cur_o
            if hasattr(self, 'bond'):
                cur_ob = 0
                for o in range(1, self.Nbonds+1):
                    if  np.all(new_o[self.bond[o]]):
                        cur_ob += 1
                        new_ob[o] = cur_ob
                Nbonds = cur_ob
            if hasattr(self, 'angle'):
                cur_oa = 0
                for o in range(1, self.Nangles+1):
                    if  np.all(new_o[self.angle[o]]):
                        cur_oa += 1
                        new_oa[o] = cur_oa
                Nangles = cur_oa
            if hasattr(self, 'dihedral'):
                cur_od = 0
                for o in range(1, self.Ndihedrals+1):
                    if  np.all(new_o[self.dihedral[o]]):
                        cur_od += 1
                        new_od[o] = cur_od
                Ndihedrals = cur_od
            if hasattr(self, 'improper'):
                cur_oi = 0
                for o in range(1, self.Nimpropers+1):
                    if  np.all(new_o[self.improper[o]]):
                        cur_oi += 1
                        new_oi[o] = cur_oi
                Nimpropers = cur_oi
            cur_om = 0
            for o in range(1, self.Nmols+1):
                if np.sum(new_o[self.mol_1st[o]: self.mol_1st[o+1]]) != 0:
                    cur_om += 1
                    new_om[o] = cur_om
        elif molfilter != None:
            molfilter = set(molfilter)
            cur_o = 0
            cur_ob = 0
            cur_oa = 0
            cur_od = 0
            cur_oi = 0
            cur_om = 0
            for o in range(1, self.Nmols+1):
                if o not in molfilter and self.mol_name[o] not in molfilter:
                    continue
                cur_om += 1
                new_om[o] = cur_om
                mol_N = self.mol_1st[o+1] - self.mol_1st[o]
                new_o[self.mol_1st[o]: self.mol_1st[o+1]] = \
                np.arange(1, mol_N+1, dtype = int) + cur_o
                cur_o += mol_N
                if hasattr(self, 'bond'):
                    mol_Nb = self.mol_b1st[o+1] - self.mol_b1st[o]
                    new_ob[self.mol_b1st[o]: self.mol_b1st[o+1]] = \
                    np.arange(1, mol_Nb+1, dtype = int) + cur_ob
                    cur_ob += mol_Nb
                if hasattr(self, 'angle'):
                    mol_Na = self.mol_a1st[o+1] - self.mol_a1st[o]
                    new_oa[self.mol_a1st[o]: self.mol_a1st[o+1]] = \
                    np.arange(1, mol_Na+1, dtype = int) + cur_oa
                    cur_oa += mol_Na
                if hasattr(self, 'dihedral'):
                    mol_Nd = self.mol_d1st[o+1] - self.mol_d1st[o]
                    new_od[self.mol_d1st[o]: self.mol_d1st[o+1]] = \
                    np.arange(1, mol_Nd+1, dtype = int) + cur_od
                    cur_od += mol_Nd
                if hasattr(self, 'improper'):
                    mol_Ni = self.mol_i1st[o+1] - self.mol_i1st[o]
                    new_oi[self.mol_i1st[o]: self.mol_i1st[o+1]] = \
                    np.arange(1, mol_Ni+1, dtype = int) + cur_oi
                    cur_oi += mol_Ni
            N = cur_o
            Nbonds = cur_ob
            Nangles = cur_oa
            Ndihedrals = cur_od
            Nimpropers = cur_oi
        if Nbonds == 0:
            chapter_present['Bonds'] = False
        if Nangles == 0:
            chapter_present['Angles'] = False
        if Ndihedrals == 0:
            chapter_present['Dihedrals'] = False
        if Nimpropers == 0:
            chapter_present['Impropers'] = False

        data = open(newdataname, 'w')

        c = self.comment.split('\n')
        c = ['# ' + l + '\n' for l in c[:-1]]
        comment = ''.join(c)
        data.write(comment)
        data.write('\n')

        data.write('%8d atoms\n' % N)
        if Nbonds > 0 :     data.write('%8d bonds\n' % Nbonds)
        if Nangles > 0 :    data.write('%8d angles\n' % Nangles)
        if Ndihedrals > 0 : data.write('%8d dihedrals\n' % Ndihedrals)
        if Nimpropers > 0 : data.write('%8d impropers\n' % Nimpropers)
        data.write('\n')
        data.write('%8d atom types\n' % self.Ntypes)
        if self.Nbtypes > 0: data.write('%8d bond types\n' % self.Nbtypes)
        if self.Natypes > 0: data.write('%8d angle types\n' % self.Natypes)
        if self.Ndtypes > 0: data.write('%8d dihedral types\n' % self.Ndtypes)
        if self.Nitypes > 0: data.write('%8d improper types\n' % self.Nitypes)
        data.write('\n')
        data.write('%12.6f %12.6f  xlo xhi\n' % (self.xlo, self.xhi))
        data.write('%12.6f %12.6f  ylo yhi\n' % (self.ylo, self.yhi))
        data.write('%12.6f %12.6f  zlo zhi\n' % (self.zlo, self.zhi))
        if self.is_tilted: data.write('%12.6f %12.6f %12.6f  xy xz yz\n' % (self.xy, self.xz, self.yz))

        for chp_name in self.chapters:
            if not chapter_present[chp_name]:
                continue

            data.write('\n' + chp_name)
            comment = self.chapter_comment[chp_name]
            if comment != '':
                data.write(' # ' + comment)
            data.write('\n\n')

            if chp_name == 'Masses':
                for o in range(1, self.Ntypes+1):
                    data.write('%8d %10.4f    #\t%s\t%s' %
                    (o, self.type_m[o], self.type_elm[o], self.type_ff[o]))
                    if self.type_text[o]:
                        data.write('\t' + self.type_text[o])
                    data.write('\n')

            elif chp_name == 'PairIJ Coeffs':
                for o0 in range(1, self.Ntypes + 1):
                    for o1 in range(o0, self.Ntypes + 1):
                        cc = self.PairIJ_Coeffs[o0, o1]
                        if not cc:
                            cc = self.PairIJ_Coeffs[o1, o0]
                        if not cc:
                            cc = ['NaN']
                            print("Warning: No PairIJ coefficients for type pair %d %d" % (o0, o1))
                        data.write('%8d%8d\t' % (o0, o1))
                        data.write('\t'.join(cc))
                        data.write('\t#\t%s\t%s\n' % (self.type_ff[o0], self.type_ff[o1]))

            elif chp_name in self.coef_chapters:
                attr_name = chp_name.replace(' ', '_')
                coeffs = getattr(self, attr_name)
                for o in range(1, len(coeffs)):
                    data.write('%8d' % o)
                    if coeffs[o] is None:
                        data.write('\tNaN')
                        print("Warning: No %s for type %d" % (chp_name, o))
                    else:
                        for c in coeffs[o]:
                            data.write('\t' + c)
                    if chp_name == 'Bond Coeffs':
                        data.write('\t# ' + str(self.btype_order[o]))
                        if self.btype_ff[o]:
                            data.write('\t' + self.btype_ff[o])
                            if self.btype_text[o]:
                                data.write('\t' + self.btype_text[o])
                    elif chp_name == 'Angle Coeffs':
                        if self.atype_ff[o]:
                            data.write('\t#\t' + self.atype_ff[o])
                            if self.atype_text[o]:
                                data.write('\t' + self.atype_text[o])
                    elif chp_name == 'Dihedral Coeffs':
                        if self.dtype_ff[o]:
                            data.write('\t#\t' + self.dtype_ff[o])
                            if self.dtype_text[o]:
                                data.write('\t' + self.dtype_text[o])
                    elif chp_name == 'Improper Coeffs':
                        if self.itype_ff[o]:
                            data.write('\t#\t' + self.itype_ff[o])
                            if self.itype_text[o]:
                                data.write('\t' + self.itype_text[o])
                    data.write('\n')

            elif chp_name == 'Atoms':
                if is_filtered:
                    for o in range(1, self.N+1):
                        if new_o[o] == 0:
                            continue
                        omol = self.omol[o]
                        data.write("%8d %7d %7d %10.6f %12.6f %12.6f %12.6f    # %s" % \
                        (new_o[o], new_om[omol], self.t[o], self.q[o],
                        self.r[o, 0], self.r[o, 1], self.r[o, 2], self.mol_name[omol]))
                        if self.text[o]:
                            data.write('\t' + self.text[o])
                        data.write('\n')
                else:
                    for o in range(1, self.N+1):
                        omol = self.omol[o]
                        data.write("%8d %7d %7d %10.6f %12.6f %12.6f %12.6f    # %s" % \
                        (o, omol, self.t[o], self.q[o],
                        self.r[o, 0], self.r[o, 1], self.r[o, 2], self.mol_name[omol]))
                        if self.text[o]:
                            data.write('\t' + self.text[o])
                        data.write('\n')

            elif chp_name =='Velocities':
                if is_filtered:
                    for o in range(1, self.N+1):
                        if new_o[o] == 0:
                            continue
                        data.write(" %8d %15.10f %15.10f %15.10f\n" % \
                        (new_o[o], self.v[o, 0], self.v[o, 1], self.v[o, 2]))
                else:
                    for o in range(1, self.N+1):
                        data.write(" %8d %15.10f %15.10f %15.10f\n" % \
                        (o, self.v[o, 0], self.v[o, 1], self.v[o, 2]))

            elif chp_name in self.link_chapters:
                is_bond = (chp_name == 'Bonds')
                link = getattr(self, chp_name[:-1].lower())
                link_t = getattr(self, chp_name[:-1].lower() + '_t')
                N = link.shape[0] - 1
                if is_filtered:
                    new_ol = new_ol_dict[chp_name]
                    for ol in range(1, N+1):
                        if new_ol[ol] == 0:
                            continue
                        data.write('%8d %7d' % (new_ol[ol], link_t[ol]))
                        for o in link[ol]:
                            data.write(' %7d' % new_o[o])
                        if  is_bond:
                            data.write('    # ' + str(self.bond_order[ol]))
                        data.write('\n')
                else:
                    for ol in range(1, N+1):
                        data.write('%8d %7d' % (ol, link_t[ol]))
                        for o in link[ol]:
                            data.write(' %7d' % o)
                        if  is_bond:
                            data.write('    # ' + str(self.bond_order[ol]))
                        data.write('\n')

        data.close()
        del data

    def write_msi(self, msiname, molfilter = None, atomfilter = None, bondthresh = 8.0):
        "Cerius2 structure file (.msi)"

        bondtypes = {1: 1, 2: 2, 3: 4, 6: 8, 7: 8}

        new_o = np.zeros(self.N+1, dtype = int)
        new_ob = np.zeros(self.Nbonds+1, dtype = int)

        cur_o = 1
        if atomfilter != None:
            atomfilter = set(atomfilter)
            for o in range(1, self.N+1):
                if o in atomfilter:
                    cur_o += 1
                    new_o[o] = cur_o
            if hasattr(self, 'bond'):
                for o in range(1, self.Nbonds+1):
                    o0, o1 = self.bond[o]
                    if new_o[o0] != 0 and new_o[o1] != 0:
                        if bondthresh != None \
                        and np.linalg.norm(self.r[o1] - self.r[o0]) > bondthresh:
                            continue
                        cur_o += 1
                        new_ob[o] = cur_o
        elif molfilter != None:
            molfilter = set(molfilter)
            for o in range(1, self.Nmols+1):
                if o not in molfilter and self.mol_name[o] not in molfilter:
                    continue
                mol_N = self.mol_1st[o+1] - self.mol_1st[o]
                new_o[self.mol_1st[o]: self.mol_1st[o+1]] = \
                np.arange(1, mol_N+1, dtype = int) + cur_o
                cur_o += mol_N
            if hasattr(self, 'bond'):
                for o in range(1, self.Nmols+1):
                    if o not in molfilter and self.mol_name[o] not in molfilter:
                        continue
                    if bondthresh == None:
                        mol_Nb = self.mol_b1st[o+1] - self.mol_b1st[o]
                        new_ob[self.mol_b1st[o]: self.mol_b1st[o+1]] = \
                        np.arange(1, mol_Nb+1, dtype = int) + cur_o
                        cur_o += mol_Nb
                    else:
                        for ob in range(self.mol_b1st[o], self.mol_b1st[o+1]):
                            o0, o1 = self.bond[ob]
                            if np.linalg.norm(self.r[o1] - self.r[o0]) <= bondthresh:
                                cur_o += 1
                                new_ob[ob] = cur_o
        else:
            new_o[1: self.N+1] = \
            np.arange(1, self.N+1, dtype = int) + cur_o
            cur_o += self.N
            if bondthresh == None:
                new_ob[1: self.Nbonds+1] = \
                np.arange(1, self.Nbonds+1, dtype = int) + cur_o
                cur_o += self.Nbonds
            else:
                for ob in range(1, self.Nbonds+1):
                    o0, o1 = self.bond[ob]
                    if np.linalg.norm(self.r[o1] - self.r[o0]) <= bondthresh:
                        cur_o += 1
                        new_ob[ob] = cur_o

        msi = open(msiname, 'w')

        msi.write("""# MSI CERIUS2 DataModel File Version 4 0
(1 Model
  (A C Label "Model")
  (A I CRY/DISPLAY (192 256))
  (A I PeriodicType 100)
  (A C SpaceGroup "1 1")
""")
        msi.write("  (A D A3 (%.4f 0 0))\n" % (self.xhi - self.xlo))
        msi.write("  (A D B3 (0 %.4f 0))\n" % (self.yhi - self.ylo))
        msi.write("  (A D C3 (0 0 %.4f))\n" % (self.zhi - self.zlo))
        msi.write("  (A D CRY/TOLERANCE 0.05)\n")

        for o in range(1, self.Nmols+1):
            if np.sum(new_o[self.mol_1st[o]: self.mol_1st[o+1]]) == 0:
                continue
            cur_o += 1
            msi.write("    (%d Subunit\n" % cur_o)
            msi.write("      (A C Label \"%s\")\n" % self.mol_name[o])
            for oatom in range(self.mol_1st[o], self.mol_1st[o+1]):
                if new_o[oatom] == 0:
                    continue
                msi.write("      (%d Atom\n" % new_o[oatom])
                otype = self.t[oatom]
                elm = self.type_elm[otype]
                msi.write("        (A C ACL \"%d %s\")\n" % (ptable.No[elm], elm))
                # msi.write("        (A C Label "O1")\n")
                msi.write("        (A D XYZ (%.6f %.6f %.6f))\n" % tuple(self.r[oatom]))
                msi.write("        (A I Id %d)\n" % oatom)
                msi.write("        (A F Charge %.6f)\n" % self.q[oatom])
                # msi.write("        (A I FChg (0 1))\n"
                msi.write("        (A C FFType \"%s\")\n" % self.type_ff[otype])
                msi.write("      )\n")
            if hasattr(self, 'bond'):
                if np.sum(new_ob[self.mol_b1st[o]: self.mol_b1st[o+1]]) == 0:
                    msi.write("    )\n")
                    continue
                for obond in range(self.mol_b1st[o], self.mol_b1st[o+1]):
                    if new_ob[obond] == 0:
                        continue
                    msi.write("      (%d Bond\n" % new_ob[obond])
                    msi.write("        (A O Atom1 %d)\n" % new_o[self.bond[obond, 0]])
                    msi.write("        (A O Atom2 %d)\n" % new_o[self.bond[obond, 1]])
                    if self.bond_order[obond] != 1:
                        # msi.write("        (A F Order %s)\n" % bond.order)
                        msi.write("        (A I Type %d)\n" % bondtypes[self.bond_order[obond]])
                    msi.write("      )\n")
            msi.write("    )\n")

        msi.write(")\n")
        msi.close()


    def finish_car(self):
        if hasattr(self, 'new_o'):
            del self.new_o
        if hasattr(self, 'new_omol'):
            del self.new_omol
        if hasattr(self, 'ff'):
            del self.ff
        if hasattr(self, 'elm'):
            del self.elm

    def write_mdf(self, mdfname, molfilter = None, atomfilter = None, bondthresh = 8.0):

        if atomfilter == None:
            if molfilter == None:
                molfilter = set(range(1, self.Nmols+1))
            elif not isinstance(molfilter, set):
                molfilter = set(molfilter)
        else:
            molfilter = set(range(1, self.Nmols+1))
            if not isinstance(atomfilter, set):
                atomfilter = set(atomfilter)

        mollist = []

        bondtypes = {2: '2.0', 3: '3.0', 6: '1.5', 7: '1.5'}

        self.new_o = np.zeros(self.N+1, dtype = int)
        new_o = self.new_o
        self.new_omol = np.zeros(self.N+1, dtype = int)
        new_omol = self.new_omol
        self.ff = self.type_ff[self.t]
        self.elm = self.type_elm[self.t]

        maxneigh = 6
        nneigh = np.zeros(self.N+1, dtype = np.uint8)
        oatoms = np.zeros((self.N+1, maxneigh), dtype = int)
        orders = np.zeros((self.N+1, maxneigh), dtype = np.uint8)

        def enlarge_arrays(oatoms, orders, maxneigh):
            new_oatoms = np.zeros((self.N+1, maxneigh+1), dtype = int)
            new_oatoms[:, :maxneigh] = oatoms
            new_orders = np.zeros((self.N+1, maxneigh+1), dtype = np.uint8)
            new_orders[:, :maxneigh] = orders
            del oatoms
            del orders
            return new_oatoms, new_orders, maxneigh+1

        cur_omol = 0
        for omol in range(1, self.Nmols+1):
            if omol not in molfilter and self.mol_name[omol] not in molfilter:
                continue
            if atomfilter != None:
                mol_N = 0
                for o in range(self.mol_1st[omol], self.mol_1st[omol+1]):
                    if o in atomfilter:
                        mol_N += 1
                        new_o[o] = mol_N
            else:
                mol_N = self.mol_1st[omol+1] - self.mol_1st[omol]
                new_o[self.mol_1st[omol]: self.mol_1st[omol+1]] = \
                np.arange(1, mol_N+1, dtype = int)
            if mol_N > 0:
                cur_omol += 1
                new_omol[self.mol_1st[omol]: self.mol_1st[omol+1]] = cur_omol
                mollist.append(omol)

        if hasattr(self, 'bond'):
            for omol in mollist:
                for ob in range(self.mol_b1st[omol], self.mol_b1st[omol+1]):
                    o0, o1 = self.bond[ob]
                    order = self.bond_order[ob]
                    if new_o[o0] != 0 and new_o[o1] != 0:
                        if bondthresh != None \
                        and np.linalg.norm(self.r[o1] - self.r[o0]) > bondthresh:
                            continue
                        try:
                            oatoms[o0, nneigh[o0]] = o1
                            oatoms[o1, nneigh[o1]] = o0
                            orders[o0, nneigh[o0]] = order
                            orders[o1, nneigh[o1]] = order
                        except IndexError:
                            oatoms, orders, maxneigh = \
                            enlarge_arrays(oatoms, orders, maxneigh)
                            oatoms[o0, nneigh[o0]] = o1
                            oatoms[o1, nneigh[o1]] = o0
                            orders[o0, nneigh[o0]] = order
                            orders[o1, nneigh[o1]] = order
                        nneigh[o0] += 1
                        nneigh[o1] += 1

        t = time.localtime(time.time())
        timestamp = '!Date: %s %s %02d %02d:%02d:%02d %d' % \
        (week_abbr[t.tm_wday], month_abbr[t.tm_mon], t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec, t.tm_year)

        mdf = open(mdfname, 'w')

        mdf.write("!BIOSYM molecular_data 4\n \n")
        mdf.write(timestamp + '   Materials Studio Generated MDF file\n')
        mdf.write("""
#topology

@column 1 element
@column 2 atom_type
@column 3 charge_group
@column 4 isotope
@column 5 formal_charge
@column 6 charge
@column 7 switching_atom
@column 8 oop_flag
@column 9 chirality_flag
@column 10 occupancy
@column 11 xray_temp_factor
@column 12 connections
""")

        for im, omol in enumerate(mollist):
            new_om = im + 1
            mdf.write(" \n@molecule " + self.mol_name[omol] + "\n \n")
            for o in range(self.mol_1st[omol], self.mol_1st[omol+1]):
                if new_o[o] == 0:
                    continue
                mdf.write("XXXX_%-14s %-2s %-7s ?     0  0   %8.4f 0 0 8 1.0000  0.0000 " % \
                (str(new_om) + ':' + self.elm[o] + str(new_o[o]), self.elm[o], self.ff[o], self.q[o]))
                for i in range(nneigh[o]):
                    o2 = oatoms[o, i]
                    mdf.write(self.elm[o2] + str(new_o[o2]))
                    if orders[o, i] != 1:
                        mdf.write('/' + bondtypes[orders[o, i]])
                    mdf.write(' ')
                mdf.write('\n')

        mdf.write("""
!
#symmetry
@periodicity 3 xyz
@group (P1)

#end
""")
        mdf.close()


    def get_car_frame(self, dr = None):

        car = io.StringIO()

        if self.is_tilted:
            a_vec = np.array([self.xhi - self.xlo, 0.0, 0.0])
            b_vec = np.array([self.xy, self.yhi - self.ylo, 0.0])
            c_vec = np.array([self.xz, self.yz, self.zhi - self.zlo])
            a = np.linalg.norm(a_vec)
            b = np.linalg.norm(b_vec)
            c = np.linalg.norm(c_vec)
            a_vec /= a; b_vec /= b; c_vec /= c
            alpha = np.rad2deg(np.arccos(np.inner(a_vec, b_vec)))
            beta  = np.rad2deg(np.arccos(np.inner(a_vec, c_vec)))
            gamma = np.rad2deg(np.arccos(np.inner(b_vec, c_vec)))
        else:
            a = self.xhi - self.xlo
            b = self.yhi - self.ylo
            c = self.zhi - self.zlo
            alpha = beta = gamma = 90.0

        if dr is None:
            dr_local = -self.com() + [a/2, b/2, c/2]
        else:
            dr_local = dr

        car.write('PBC %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f (P1)\n' % \
        (a, b, c, alpha, beta, gamma))

        cur_omol = 1
        for o in range(1, self.N+1):
            new_o = self.new_o[o]
            if new_o == 0:
                continue
            new_omol = self.new_omol[o]
            if new_omol != cur_omol:
                car.write('end\n')
                cur_omol = new_omol
            r = self.r[o] + dr_local
            car.write('%-5s %14.9f %14.9f %14.9f XXXX %-6d %-7s %-2s %6.3f\n' % \
            (self.elm[o] + str(new_o), r[0], r[1], r[2], \
            new_omol, self.ff[o], self.elm[o], self.q[o]))

        car.write('end\n')
        car.write('end\n')

        return car.getvalue()

    def write_car(self, carname, dr = None):

        t = time.localtime(time.time())
        timestamp = '!DATE %s %s %02d %02d:%02d:%02d %d\n' % \
        (week_abbr[t.tm_wday], month_abbr[t.tm_mon], t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec, t.tm_year)

        car = open(carname, 'w')

        car.write('!BIOSYM archive 3\nPBC=ON\nMaterials Studio Generated CAR File\n')
        car.write(timestamp)
        car.write(self.get_car_frame(dr))

        car.close()

    def write_arc(self, arcname, digest_generator, step = 1.0, dr = None):

        clo_original = (self.xlo, self.ylo, self.zlo)
        chi_original = (self.xhi, self.yhi, self.zhi)
        if self.is_tilted:
            tilt_original = (self.xy, self.xz, self.yz)
        r_original = np.copy(self.r)

        t = time.localtime(time.time())
        timestamp = '!DATE %s %s %02d %02d:%02d:%02d %d\n' % \
        (week_abbr[t.tm_wday], month_abbr[t.tm_mon], t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec, t.tm_year)

        arc = open(arcname, 'w')
        arc.write('!BIOSYM archive 3\nPBC=ON\n')
        arc.close()

        oframe = 0
        for f in digest_generator:
            self.xlo = f.xlo; self.xhi = f.xhi
            self.ylo = f.ylo; self.yhi = f.yhi
            self.zlo = f.zlo; self.zhi = f.zhi
            if hasattr(f, 'xy'):
                self.xy = f.xy
                self.xz = f.xz
                self.yz = f.yz
            if f.is_sparse:
                self.r[f.atomlist] = f.r
            else:
                self.r = f.r
            ps = f.timestep * step / 1000.0
            arc = open(arcname, 'a')
            arc.write('Frame %d  Number of Picoseconds = %10.4f\n' % (oframe, ps))
            oframe += 1
            arc.write(timestamp)
            arc.write(self.get_car_frame(dr))
            arc.close()

        self.xlo, self.ylo, self.zlo = clo_original
        self.xhi, self.yhi, self.zhi = chi_original
        if self.is_tilted:
            self.xy, self.xz, self.yz = tilt_original
        self.r = r_original


    def get_neighbors(self):
        """
        Return nneigh, the array of numbers of bound atoms
        and oatoms, the array of ordinals of bound atoms
        """
        maxneigh = 6
        nneigh = np.zeros(self.N+1, dtype = np.uint8)
        oatoms = np.zeros((self.N+1, maxneigh), dtype = int)
        for o0, o1 in self.bond:
            try:
                oatoms[o0, nneigh[o0]] = o1
                oatoms[o1, nneigh[o1]] = o0
            except IndexError:
                new_oatoms = np.zeros((self.N+1, maxneigh+1), dtype = int)
                new_oatoms[:, :maxneigh] = oatoms
                del oatoms
                oatoms = new_oatoms
                maxneigh += 1
                oatoms[o0, nneigh[o0]] = o1
                oatoms[o1, nneigh[o1]] = o0
            nneigh[o0] += 1
            nneigh[o1] += 1

        return nneigh, oatoms

    def get_incidence(self):
        """
        Return nneigh, the array of numbers of bound atoms,
        oatoms, the array of ordinals of bound atoms,
        and obonds, the array of ordinals of incident bonds
        """
        maxneigh = 6
        nneigh = np.zeros(self.N+1, dtype = np.uint8)
        oatoms = np.zeros((self.N+1, maxneigh), dtype = int)
        obonds = np.zeros((self.N+1, maxneigh), dtype = int)
        for ob in range(1, self.Nbonds+1):
            o0, o1 = self.bond[ob]
            try:
                oatoms[o0, nneigh[o0]] = o1
                oatoms[o1, nneigh[o1]] = o0
                obonds[o0, nneigh[o0]] = ob
                obonds[o1, nneigh[o1]] = ob
            except IndexError:
                new_oatoms = np.zeros((self.N+1, maxneigh+1), dtype = int)
                new_oatoms[:, :maxneigh] = oatoms
                new_obonds = np.zeros((self.N+1, maxneigh+1), dtype = int)
                new_obonds[:, :maxneigh] = obonds
                del oatoms
                del obonds
                oatoms = new_oatoms
                obonds = new_obonds
                maxneigh += 1
                oatoms[o0, nneigh[o0]] = o1
                oatoms[o1, nneigh[o1]] = o0
                obonds[o0, nneigh[o0]] = ob
                obonds[o1, nneigh[o1]] = ob
            nneigh[o0] += 1
            nneigh[o1] += 1

        return nneigh, oatoms, obonds

    def add_complex_links(self):
        """
        Construct the full set of angles, dihedrals and impropers
        from bonds. All of them have type 0.
        """

        nneigh, oatoms = self.get_neighbors()

        angles = [[0, 0, 0]]
        mol_nang = np.zeros(self.Nmols+1, dtype = int)
        for o in range(1, self.N+1):
            if nneigh[o] < 2: continue
            nang = 0
            for i in range(nneigh[o]):
                for j in range(i+1, nneigh[o]):
                    nang += 1
                    angles.append([oatoms[o, i], o, oatoms[o, j]])
            mol_nang[self.omol[o]] += nang
        Nangles = np.sum(mol_nang)
        if Nangles > 0:
            self.Nangles = Nangles
            self.chapter_present['Angles'] = True
            self.angle = np.array(angles, dtype = int)
            self.angle_t = np.zeros(Nangles+1, dtype = int)
            self.mol_a1st = np.zeros(self.Nmols + 2, dtype = int)
            cur_a1st = 1
            for omol in range(1, self.Nmols+2):
                cur_a1st += mol_nang[omol-1]
                self.mol_a1st[omol] = cur_a1st

        # ol0-om0-om1-ol1
        # m - medial, l - lateral
        dihedrals = [[0, 0, 0, 0]]
        mol_ndih = np.zeros(self.Nmols+1, dtype = int)
        for om0, om1 in self.bonds[1:]:
            ndih = (nneigh[om0] - 1) * (nneigh[om1] - 1)
            if ndih == 0:
                continue
            for ol0 in oatoms[om0]:
                if ol0 == om1: continue
                for ol1 in oatoms[om1]:
                    if ol1 == om0: continue
                    dihedrals.append([ol0, om0, om1, ol1])
            mol_ndih[self.omol[om0]] += ndih
        Ndihedrals = np.sum(mol_ndih)
        if Ndihedrals > 0:
            self.Ndihedrals = Ndihedrals
            self.chapter_present['Dihedrals'] = True
            self.dihedral = np.array(dihedrals, dtype = int)
            self.dihedral_t = np.zeros(Ndihedrals+1, dtype = int)
            self.mol_d1st = np.zeros(self.Nmols + 2, dtype = int)
            cur_d1st = 1
            for omol in range(1, self.Nmols+2):
                cur_d1st += mol_ndih[omol-1]
                self.mol_d1st[omol] = cur_d1st

        impropers = [[0, 0, 0, 0]]
        mol_nimp = np.zeros(self.Nmols+1, dtype = int)
        for o in range(1, self.N+1):
            if nneigh[o] != 3: continue
            impropers.append([o] + list(oatoms[o, :3]))
            mol_nimp[self.omol[o]] += 1
        Nimpropers = np.sum(mol_nimp)
        if Nimpropers > 0:
            self.Nimpropers = Nimpropers
            self.chapter_present['Impropers'] = True
            self.improper = np.array(impropers, dtype = int)
            self.improper_t = np.zeros(Nimpropers+1, dtype = int)
            self.mol_i1st = np.zeros(self.Nmols + 2, dtype = int)
            cur_i1st = 1
            for omol in range(1, self.Nmols+2):
                cur_i1st += mol_nimp[omol-1]
                self.mol_i1st[omol] = cur_i1st

    def apply_ff(self, ffname, ffparams = {}, embed_all_params = False):

        warn_ret = ''

        ff_module = import_module(ffname)
        ff = ff_module.c_ff(ffparams)

        if hasattr(ff, 'FF_Type'):
            ff.FF_Type(self, embed_all_params)
            self.chapter_present['Masses'] = True
        else:
            absent = []
            for ot in range(1, self.Ntypes+1):
                if not hasattr(ff, 'ff_mass'):
                    raise Exception('\nFatal: Forcefield definition lacks Masses section')
                try:
                    self.type_m[ot] = ff.ff_mass[self.type_ff[ot]]
                except KeyError:
                    absent.append(self.type_ff[ot])
            if len(absent) > 0:
                errortext = '\nFatal: Following atom types are absent from FF definition:'
                for fftype in absent:
                    errortext += '\n' + fftype
                raise Exception(errortext)
            if embed_all_params:
                new_Ntypes = len(ff.ff_mass)
                new_type_ff = [None] + list(ff.ff_mass.keys())
                renumerate = {ot: new_type_ff.index(self.type_ff[ot]) for ot in range(1, self.Ntypes+1)}
                for o in range(1, self.N + 1):
                    self.t[o] = renumerate[self.t[o]]
                self.type_ff = np.array(new_type_ff, dtype = 'U8')
                self.type_m = np.array([0.0] + list(ff.ff_mass.values()))
                new_type_elm = np.full(new_Ntypes+1, 'X', dtype = 'U2')
                new_type_text = np.empty(new_Ntypes+1, dtype = object)
                for ot in range(1, self.Ntypes+1):
                    new_type_elm[renumerate[ot]] = self.type_elm[ot]
                    new_type_text[renumerate[ot]] = self.type_text[ot]
                self.type_elm = new_type_elm
                self.type_text = new_type_text
                self.Ntypes = new_Ntypes
            self.chapter_present['Masses'] = True

        if hasattr(ff, 'FF_Pair'):
            absent = []
            self.Pair_Coeffs = np.empty(self.Ntypes+1, dtype = object)
            for ot in range(1, self.Ntypes+1):
                coefs = ff.FF_Pair(self.type_ff[ot])
                if coefs == []:
                    absent.append(self.type_ff[ot])
                else:
                    coefs = [str(c) for c in coefs]
                    self.Pair_Coeffs[ot] = coefs
            if len(absent) > 0:
                errortext = '\nFatal: Pair coefficients for the following types are absent from FF definition:'
                for fftype in absent:
                    errortext += '\n' + fftype
                raise Exception(errortext)
            self.chapter_present['Pair Coeffs'] = True

##            if hasattr(ff, 'ff_pair_style'):
##                self.chapter_comment['Pair Coeffs'] = ff.ff_pair_style

        elif hasattr(ff, 'FF_PairIJ'):
            absent = []
            self.PairIJ_Coeffs = np.empty((self.Ntypes+1, self.Ntypes+1), dtype = object)
            for i in range(1, self.Ntypes+1):
                for j in range(1, i+1):
                    coefs = ff.FF_PairIJ(self.type_ff[i], self.type_ff[j])
                    if coefs == []:
                        ff_key0 = '-'.join([self.atom_types[i].fftype, self.atom_types[j].fftype])
                        ff_key1 = '-'.join([self.atom_types[j].fftype, self.atom_types[i].fftype])
                        if not (ff_key0 in absent or ff_key1 in absent):
                            absent.append(ff_key0)
                    else:
                        coefs = [str(c) for c in coefs]
                        self.PairIJ_Coeffs[i, j] = coefs
                        self.PairIJ_Coeffs[j, i] = coefs
            if len(absent) > 0:
                errortext = '\nFatal: Pair coefficients for the following type pairs are absent from FF definition:'
                for ff_key in absent:
                    errortext += '\n' + ff_key
                raise Exception(errortext)
            self.chapter_present['PairIJ Coeffs'] = True

##            if hasattr(ff, 'ff_pair_style'):
##                self.chapter_comment['PairIJ Coeffs'] = ff.ff_pair_style

        nneigh, oatoms, obonds = self.get_incidence()
        self.ff = self.type_ff[self.t]

        if hasattr(ff, 'FF_Charge'):
            ff.FF_Charge(self)
        elif hasattr(ff, 'FF_Bond_Increment'):
            self.q[:] = 0.0
            absent = []
            for ob in range(1, self.Nbonds + 1):
                o0, o1 = self.bond[ob]
                res = ff.FF_Bond_Increment(self, o0, ob, o1)
                if res == None:
                    ff_key0 = '-'.join([self.ff[o0], self.ff[o1]])
                    ff_key1 = '-'.join([self.ff[o1], self.ff[o0]])
                    if not (ff_key0 in absent or ff_key1 in absent):
                        absent.append(ff_key0)
                else:
                    increment = res
                    self.q[o0] += increment
                    self.q[o1] -= increment

            if len(absent) > 0:
                errortext = '\nFatal: Bond increments of charge for the following type pairs are absent from FF definition:'
                for ff_key in absent:
                    errortext += '\n' + ff_key
                raise Exception(errortext)

        if hasattr(ff, 'FF_Bond'):
            absent = []
            present = [None]
            if embed_all_params and hasattr(ff, 'ff_bond'):
                present += list(ff.ff_bond.keys())
            for ob in range(1, self.Nbonds + 1):
                o0, o1 = self.bond[ob]
                res = ff.FF_Bond(self, o0, ob, o1)
                if res == None:
                    ff_key0 = '-'.join([self.ff[o0], self.ff[o1]])
                    ff_key1 = '-'.join([self.ff[o1], self.ff[o0]])
                    if not (ff_key0 in absent or ff_key1 in absent):
                        absent.append(ff_key0)
                else:
                    ff_key, bond = res
                    if ff_key not in present:
                        present.append(ff_key)
                    self.bond_t[ob] = present.index(ff_key)
                    self.bond[ob] = bond

            if len(absent) > 0:
                errortext = '\nFatal: Bond coefficients for the following type pairs are absent from FF definition:'
                for ff_key in absent:
                    errortext += '\n' + ff_key
                raise Exception(errortext)

            self.Nbtypes = len(present) - 1
            self.btype_ff = np.array(present, dtype = object)
            self.btype_text = np.empty_like(self.btype_ff)
            self.btype_order = np.zeros(self.Nbtypes+1, dtype = np.uint8)
            self.Bond_Coeffs = np.empty(self.Nbtypes+1, dtype = object)
            for obt in range(1, self.Nbtypes+1):
                ff_key = self.btype_ff[obt]
                coef_dict = ff.ff_bond[ff_key]
                coefs = [str(c) for c in coef_dict['Bond_Coeffs']]
                self.Bond_Coeffs[obt] = coefs
                self.btype_order[obt] = coef_dict.get('Order', 1)
            self.chapter_present['Bond Coeffs'] = True

##            if hasattr(ff, 'ff_bond_style'):
##                self.chapter_comment['Bond Coeffs'] = ff.ff_bond_style


        if hasattr(ff, 'FF_Angle'):
            angles = [[0, 0, 0]]
            angle_t = [0]
            mol_nang = np.zeros(self.Nmols+1, dtype = int)
            absent = []
            present = {}
            if embed_all_params and hasattr(ff, 'ff_angle'):
                present = {ff_key: o + 1 for o, ff_key in enumerate(ff.ff_angle.keys())}
            for o in range(1, self.N+1):
                if nneigh[o] < 2: continue
                for i in range(nneigh[o]):
                    for j in range(i+1, nneigh[o]):
                        ol0 = oatoms[o, i]; ol1 = oatoms[o, j]
                        ob0 = obonds[o, i]; ob1 = obonds[o, j]
                        matches = ff.FF_Angle(self, ol0, ob0, o, ob1, ol1)
                        if len(matches) == 0:
                            fftypel0 = self.ff[ol0]
                            fftypec0 = self.ff[o]
                            fftypel1 = self.ff[ol1]
                            ff_key0 = '-'.join([fftypel0, fftypec0, fftypel1])
                            ff_key1 = '-'.join([fftypel1, fftypec0, fftypel0])
                            if not (ff_key0 in absent or ff_key1 in absent):
                                absent.append(ff_key0)
                        else:
                            for ff_key, angle in matches:
                                if ff_key not in present:
                                    otype = len(present) + 1
                                    present[ff_key] = otype
                                angles.append(angle)
                                angle_t.append(present[ff_key])
                                mol_nang[self.omol[o]] += 1

            if len(absent) > 0:
                errortext = '\nWarning: Angle coefficients for the following type triples are absent from FF definition:'
                for ff_key in absent:
                    errortext += '\n' + ff_key
                warn_ret += errortext

            self.Nangles = np.sum(mol_nang)
            if self.Nangles > 0:
                self.chapter_present['Angles'] = True
                self.angle = np.array(angles)
                self.angle_t = np.array(angle_t)
                self.Natypes = len(present)
                self.atype_ff = np.empty(self.Natypes+1, dtype = object)
                self.atype_text = np.empty(self.Natypes+1, dtype = object)
                for ff_key in ff.ff_angle:
                    coef_dict = ff.ff_angle[ff_key]
                    for coef_name in coef_dict:
                        setattr(self, coef_name, np.empty(self.Natypes+1, dtype = object))
                        self.chapter_present[coef_name.replace('_', ' ')] = True
                    break
                for ff_key in present:
                    oat = present[ff_key]
                    self.atype_ff[oat] = ff_key
                    coef_dict = ff.ff_angle[ff_key]
                    for coef_name in coef_dict:
                        coefs = [str(c) for c in coef_dict[coef_name]]
                        coef_array = getattr(self, coef_name)
                        coef_array[oat] = coefs
                self.mol_a1st = np.zeros(self.Nmols + 2, dtype = int)
                cur_a1st = 1
                for omol in range(1, self.Nmols+2):
                    cur_a1st += mol_nang[omol-1]
                    self.mol_a1st[omol] = cur_a1st

##                if hasattr(ff, 'ff_angle_style'):
##                    self.chapter_comment['Angle Coeffs'] = ff.ff_angle_style


        if hasattr(ff, 'FF_Dihedral'):
            dihedrals = [[0, 0, 0, 0]]
            dihedral_t = [0]
            mol_ndih = np.zeros(self.Nmols+1, dtype = int)
            absent = []
            present = {}
            if embed_all_params and hasattr(ff, 'ff_dihedral'):
                present = {ff_key: o + 1 for o, ff_key in enumerate(ff.ff_dihedral.keys())}
            for ob in range(1, self.Nbonds+1):
                om0, om1 = self.bond[ob]
                if nneigh[om0] < 2 or nneigh[om1] < 2:
                    continue
                for i in range(nneigh[om0]):
                    ol0 = oatoms[om0, i]
                    ob0 = obonds[om0, i]
                    if ol0 == om1: continue
                    for j in range(nneigh[om1]):
                        ol1 = oatoms[om1, j]
                        ob1 = obonds[om1, j]
                        if ol1 == om0: continue
                        matches = ff.FF_Dihedral(self, ol0, ob0, om0, ob, om1, ob1, ol1)
                        if len(matches) == 0:
                            fftypel0 = self.ff[ol0]
                            fftypem0 = self.ff[om0]
                            fftypem1 = self.ff[om1]
                            fftypel1 = self.ff[ol1]
                            ff_key0 = '-'.join([fftypel0, fftypem0, fftypem1, fftypel1])
                            ff_key1 = '-'.join([fftypel1, fftypem1, fftypem0, fftypel0])
                            if not (ff_key0 in absent or ff_key1 in absent):
                                absent.append(ff_key0)
                        else:
                            for ff_key, dihedral in matches:
                                if ff_key not in present:
                                    otype = len(present) + 1
                                    present[ff_key] = otype
                                dihedrals.append(dihedral)
                                dihedral_t.append(present[ff_key])
                                mol_ndih[self.omol[om0]] += 1

            if len(absent) > 0:
                errortext = '\nWarning: Dihedral coefficients for the following type tuples are absent from FF definition:'
                for ff_key in absent:
                    errortext += '\n' + ff_key
                warn_ret += errortext

            self.Ndihedrals = np.sum(mol_ndih)
            if self.Ndihedrals > 0:
                self.chapter_present['Dihedrals'] = True
                self.dihedral = np.array(dihedrals)
                self.dihedral_t = np.array(dihedral_t)
                self.Ndtypes = len(present)
                self.dtype_ff = np.empty(self.Ndtypes+1, dtype = object)
                self.dtype_text = np.empty(self.Ndtypes+1, dtype = object)
                for ff_key in ff.ff_dihedral:
                    coef_dict = ff.ff_dihedral[ff_key]
                    for coef_name in coef_dict:
                        setattr(self, coef_name, np.empty(self.Ndtypes+1, dtype = object))
                        self.chapter_present[coef_name.replace('_', ' ')] = True
                    break
                for ff_key in present:
                    odt = present[ff_key]
                    self.dtype_ff[odt] = ff_key
                    coef_dict = ff.ff_dihedral[ff_key]
                    for coef_name in coef_dict:
                        coefs = [str(c) for c in coef_dict[coef_name]]
                        coef_array = getattr(self, coef_name)
                        coef_array[odt] = coefs
                self.mol_d1st = np.zeros(self.Nmols + 2, dtype = int)
                cur_d1st = 1
                for omol in range(1, self.Nmols+2):
                    cur_d1st += mol_ndih[omol-1]
                    self.mol_d1st[omol] = cur_d1st

##                if hasattr(ff, 'ff_dihedral_style'):
##                    self.chapter_comment['Dihedral Coeffs'] = ff.ff_dihedral_style


        def improper_keys(tc0, tl0, tl1, tl2):
            yield tc0 +'-'+ tl0 +',-'+ tl1 +',-'+ tl2
            yield tc0 +'-'+ tl0 +',-'+ tl2 +',-'+ tl1
            yield tc0 +'-'+ tl1 +',-'+ tl0 +',-'+ tl2
            yield tc0 +'-'+ tl2 +',-'+ tl0 +',-'+ tl1
            yield tc0 +'-'+ tl1 +',-'+ tl2 +',-'+ tl0
            yield tc0 +'-'+ tl2 +',-'+ tl1 +',-'+ tl0

        if hasattr(ff, 'FF_Improper'):
            impropers = [[0, 0, 0, 0]]
            improper_t = [0]
            mol_nimp = np.zeros(self.Nmols+1, dtype = int)
            absent_true = []
            absent_false = []
            present = {}
            if embed_all_params and hasattr(ff, 'ff_improper'):
                present = {ff_key: o + 1 for o, ff_key in enumerate(ff.ff_improper.keys())}
            for o in range(1, self.N+1):
                if nneigh[o] < 3: continue
                elif nneigh[o] == 3:
                    absent = absent_true
                else:
                    absent = absent_false
                for i in range(nneigh[o]):
                    for j in range(i+1, nneigh[o]):
                        for k in range(j+1, nneigh[o]):
                            ol0 = oatoms[o, i]; ol1 = oatoms[o, j]; ol2 = oatoms[o, k]
                            ob0 = obonds[o, i]; ob1 = obonds[o, j]; ob2 = obonds[o, k]
                            matches = ff.FF_Improper(self, o, ob0, ol0, ob1, ol1, ob2, ol2)
                            if len(matches) == 0:
                                fftypec0 = self.ff[o]
                                fftypel0 = self.ff[ol0]
                                fftypel1 = self.ff[ol1]
                                fftypel2 = self.ff[ol2]
                                not_in_absent = True
                                for ff_key in improper_keys(fftypec0, fftypel0, fftypel1, fftypel2):
                                    if ff_key in absent:
                                        not_in_absent = False
                                if not_in_absent:
                                    absent.append(ff_key)
                            else:
                                for ff_key, improper in matches:
                                    if ff_key not in present:
                                        otype = len(present) + 1
                                        present[ff_key] = otype
                                    impropers.append(improper)
                                    improper_t.append(present[ff_key])
                                    mol_nimp[self.omol[o]] += 1

            if len(absent_true) > 0:
                errortext = '\nWarning: "True" improper coefficients for the following type tuples are absent from FF definition:'
                for ff_key in absent_true:
                    errortext += '\n' + ff_key
                warn_ret += errortext
            if len(absent_false) > 0:
                errortext = '\nWarning: "False" improper coefficients for the following type tuples are absent from FF definition:'
                for ff_key in absent_false:
                    errortext += '\n' + ff_key
                warn_ret += errortext

            self.Nimpropers = np.sum(mol_nimp)
            if self.Nimpropers > 0:
                self.chapter_present['Impropers'] = True
                self.improper = np.array(impropers)
                self.improper_t = np.array(improper_t)
                self.Nitypes = len(present)
                self.itype_ff = np.empty(self.Nitypes+1, dtype = object)
                self.itype_text = np.empty(self.Nitypes+1, dtype = object)
                for ff_key in ff.ff_improper:
                    coef_dict = ff.ff_improper[ff_key]
                    for coef_name in coef_dict:
                        setattr(self, coef_name, np.empty(self.Nitypes+1, dtype = object))
                        self.chapter_present[coef_name.replace('_', ' ')] = True
                    break
                for ff_key in present:
                    oit = present[ff_key]
                    self.itype_ff[oit] = ff_key
                    coef_dict = ff.ff_improper[ff_key]
                    for coef_name in coef_dict:
                        coefs = [str(c) for c in coef_dict[coef_name]]
                        coef_array = getattr(self, coef_name)
                        coef_array[oit] = coefs
                self.mol_i1st = np.zeros(self.Nmols + 2, dtype = int)
                cur_i1st = 1
                for omol in range(1, self.Nmols+2):
                    cur_i1st += mol_nimp[omol-1]
                    self.mol_i1st[omol] = cur_i1st

##                if hasattr(ff, 'ff_improper_style'):
##                    self.chapter_comment['Improper Coeffs'] = ff.ff_improper_style


        del self.ff

        return warn_ret

def unite(cells, stub = None, t_dicts = None, bt_dicts = None, at_dicts = None, dt_dicts = None, it_dicts = None):
    """Make one cell from several ones.
    cells is a container with c_cell objects or .data file names;
    stub is a c_cell or .data with no atoms but cell dimensions,
    atom types and all interaction coefficients determined; if absent
    all these from the zeroth element of cells are used;
    Xt_dicts is a container with dictionaries to convert atom, bonc, etc.
    types from respective cells element to stub; some Xt_dicts elements
    may be None"""
    
    if stub != None:
        retcell = stub
    else:
        retcell = deepcopy(cells[0])
    
    for type_dicts, attr in [(t_dicts, 't'), (bt_dicts, 'bond_t'), (at_dicts, 'angle_t'), 
    (dt_dicts, 'dihedral_t'), (it_dicts, 'improper_t')]:
        if type_dicts != None:
            for t_d, cell in zip(t_dicts, cells):
                if t_d != None and hasattr(cell, attr):
                    Nt = max(t_d)
                    mapping = np.zeros(Nt + 1, dtype = int)
                    for t0 in t_d:
                        mapping[t0] = t_d[t0]
                    t = getattr(cell, attr)
                    setattr(cell, 'new_' + attr, mapping[t])
    
    retcell.N = sum([cell.N for cell in cells])
    retcell.Nbonds = sum([cell.Nbonds for cell in cells])
    retcell.Nangles = sum([cell.Nangles for cell in cells])
    retcell.Ndihedrals = sum([cell.Ndihedrals for cell in cells])
    retcell.Nimpropers = sum([cell.Nimpropers for cell in cells])
    retcell.Nmols = sum([cell.Nmols for cell in cells])

    retcell.chapter_present['Masses'] = True
    retcell.chapter_present['Atoms'] = True
    retcell.chapter_present['Velocities'] = any([cell.chapter_present['Velocities'] for cell in cells])

    atombase = 0
    molbase = 0
    Lnew_mol_1st = []
    Lnew_omol = []
    for cell in cells:
        new_mol_1st = cell.mol_1st + atombase
        new_omol = cell.omol + molbase
        atombase += cell.N
        molbase += cell.Nmols
        Lnew_mol_1st.append(new_mol_1st)
        Lnew_omol.append(new_omol)
    retcell.r = np.concatenate([cells[0].r] + [cell.r[1:] for cell in cells[1:]])
    if retcell.chapter_present['Velocities']:
        for cell in cells:
            if not cell.chapter_present['Velocities']:
                cell.v = np.zeros_like(cell.r)
        retcell.v = np.concatenate([cells[0].v] + [cell.v[1:] for cell in cells[1:]])
    retcell.q = np.concatenate([cells[0].q] + [cell.q[1:] for cell in cells[1:]])
    retcell.t = np.concatenate([cells[0].new_t] + [cell.new_t[1:] for cell in cells[1:]])
    retcell.text = np.concatenate([cells[0].text] + [cell.text[1:] for cell in cells[1:]])
    retcell.omol = np.concatenate([Lnew_omol[0]] + [no[1:] for no in Lnew_omol[1:]])
    retcell.mol_1st = np.concatenate([Lnew_mol_1st[0][:-1]] + [nm1[1:-1] for nm1 in Lnew_mol_1st[1:-1]] \
    + [Lnew_mol_1st[-1][1:]])
    Ndigits = len(str(len(cells) + 1))
    for ic, cell in enumerate(cells):
        cell.new_mol_name = np.zeros_like(cell.mol_name)
        for omol in range(1, cell.Nmols + 1):
            cell.new_mol_name[omol] = str(ic + 1).zfill(Ndigits) + '_' + cell.mol_name[omol]
    retcell.mol_name = np.concatenate([cells[0].new_mol_name] + [ch.new_mol_name[1:] for ch in cells[1:]])

    if any([cell.chapter_present['Bonds'] for cell in cells]):
        retcell.chapter_present['Bonds'] = True
        atombase = 0; bondbase = 0
        Lnew_bond = []; Lnew_bond_t = []
        Lbond_order = []; Lnew_mol_b1st = []
        for cell in cells:
            if cell.chapter_present['Bonds']:
                new_bond = cell.bond + atombase
                new_mol_b1st = cell.mol_b1st + bondbase
                atombase += cell.N
                bondbase += cell.Nbonds
                Lnew_bond.append(new_bond)
                Lnew_bond_t.append(cell.new_bond_t)
                Lbond_order.append(cell.bond_order)
                Lnew_mol_b1st.append(new_mol_b1st)
        retcell.bond = np.concatenate([Lnew_bond[0]] + [nb[1:] for nb in Lnew_bond[1:]])
        retcell.bond_t = np.concatenate([Lnew_bond_t[0]] + [nbt[1:] for nbt in Lnew_bond_t[1:]])
        retcell.bond_order = np.concatenate([Lbond_order[0]] + [bo[1:] for bo in Lbond_order[1:]])
        if len(Lnew_mol_b1st) > 1:
            retcell.mol_b1st = np.concatenate([Lnew_mol_b1st[0][:-1]] \
            + [nmb1[1:-1] for nmb1 in Lnew_mol_b1st[1:-1]] + [Lnew_mol_b1st[-1][1:]])
        else:
            retcell.mol_b1st = Lnew_mol_b1st[0]

    if any([cell.chapter_present['Angles'] for cell in cells]):
        retcell.chapter_present['Angles'] = True
        atombase = 0; anglebase = 0
        Lnew_angle = []; Lnew_angle_t = []
        Lnew_mol_a1st = []
        for cell in cells:
            if cell.chapter_present['Angles']:
                new_angle = cell.angle + atombase
                new_mol_a1st = cell.mol_a1st + anglebase
                atombase += cell.N
                anglebase += cell.Nangles
                Lnew_angle.append(new_angle)
                Lnew_angle_t.append(cell.new_angle_t)
                Lnew_mol_a1st.append(new_mol_a1st)
        retcell.angle = np.concatenate([Lnew_angle[0]] + [na[1:] for na in Lnew_angle[1:]])
        retcell.angle_t = np.concatenate([Lnew_angle_t[0]] + [nat[1:] for nat in Lnew_angle_t[1:]])
        if len(Lnew_mol_a1st) > 1:
            retcell.mol_a1st = np.concatenate([Lnew_mol_a1st[0][:-1]] \
            + [nma1[1:-1] for nma1 in Lnew_mol_a1st[1:-1]] + [Lnew_mol_a1st[-1][1:]])
        else:
            retcell.mol_a1st = Lnew_mol_a1st[0]

    if any([cell.chapter_present['Dihedrals'] for cell in cells]):
        retcell.chapter_present['Dihedrals'] = True
        atombase = 0; dihedralbase = 0
        Lnew_dihedral = []; Lnew_dihedral_t = []
        Lnew_mol_d1st = []
        for cell in cells:
            if cell.chapter_present['Dihedrals']:
                new_dihedral = cell.dihedral + atombase
                new_mol_d1st = cell.mol_d1st + dihedralbase
                atombase += cell.N
                dihedralbase += cell.Ndihedrals
                Lnew_dihedral.append(new_dihedral)
                Lnew_dihedral_t.append(cell.new_dihedral_t)
                Lnew_mol_d1st.append(new_mol_d1st)
        retcell.dihedral = np.concatenate([Lnew_dihedral[0]] + [nd[1:] for nd in Lnew_dihedral[1:]])
        retcell.dihedral_t = np.concatenate([Lnew_dihedral_t[0]] + [ndt[1:] for ndt in Lnew_dihedral_t[1:]])
        if len(Lnew_mol_d1st) > 1:
            retcell.mol_d1st = np.concatenate([Lnew_mol_d1st[0][:-1]] \
            + [nmd1[1:-1] for nmd1 in Lnew_mol_d1st[1:-1]] + [Lnew_mol_d1st[-1][1:]])
        else:
            retcell.mol_d1st = Lnew_mol_d1st[0]

    if any([cell.chapter_present['Impropers'] for cell in cells]):
        retcell.chapter_present['Impropers'] = True
        atombase = 0; improperbase = 0
        Lnew_improper = []
        Lnew_mol_i1st = []
        for cell in cells:
            if cell.chapter_present['Impropers']:
                new_improper = cell.improper + atombase
                new_mol_i1st = cell.mol_i1st + improperbase
                atombase += cell.N
                improperbase += cell.Nimpropers
                Lnew_improper.append(new_improper)
                Lnew_mol_i1st.append(new_mol_i1st)
        retcell.improper = np.concatenate([Lnew_improper[0]] + [ni[1:] for ni in Lnew_improper[1:]])
        if len(Lnew_mol_i1st) > 1:
            retcell.mol_i1st = np.concatenate([Lnew_mol_i1st[0][:-1]] \
            + [nmi1[1:-1] for nmi1 in Lnew_mol_i1st[1:-1]] + [Lnew_mol_i1st[-1][1:]])
        else:
            retcell.mol_i1st = Lnew_mol_i1st[0]

    for cell in cells:
        for attr in ['new_t', 'new_bond_t', 'new_angle_t', 'new_dihedral_t', 'new_improper_t']:
            if hasattr(cell, attr):
                delattr(cell, attr)
        del cell.new_mol_name

    return retcell
    



def add_solvent(forcefield, cell, solv, N, clo = None, chi = None, Nrot = 10000, Nshift = 1000):

    if cell:
        if clo is None or chi is None:
            clo = np.array([cell.xlo, cell.ylo, cell.zlo])
            chi = np.array([cell.xhi, cell.yhi, cell.zhi])
    else:
        if clo is None or chi is None:
            raise ValueError()

    size = chi - clo

    solv.orient()

    ff_module = import_module(forcefield)
    ff = ff_module.c_ff({})

    type_hsigma = np.zeros(solv.Ntypes+1)       # half Lennard-Jones' sigma for each force field type
    mult = (2**(1./6.)) / 2
    for ot in range(1, solv.Ntypes+1):
        type_hsigma[ot] = ff.ff_pair[solv.type_ff[ot]][-1] * mult
    hsigma = type_hsigma[solv.t[1:]]
    sigma_max = np.max(hsigma) * 2

    if cell:
        cell.type_hsigma = np.zeros(cell.Ntypes+1)
        for ot in range(1, cell.Ntypes+1):
            cell.type_hsigma[ot] = ff.ff_pair[cell.type_ff[ot]][-1] * mult
        cell.hsigma = cell.type_hsigma[cell.t[1:]]
        cell.sigma_max = np.max(cell.hsigma) * 2
        sigma_max = max(sigma_max, cell.sigma_max)

    Ntiles = (size / sigma_max).astype(int)
    DIRECTORY = np.empty(Ntiles, dtype = object)
    random_pa = random_polar_angle()

    cells = []
    if cell:
        cells = [cell]
        room_test_pbc(cell.r[1:], cell.hsigma, DIRECTORY, size)
    rot_r = np.empty((solv.N, 3))
    shift_r = np.empty((solv.N, 3))
    for i in range(N):
        trial = deepcopy(solv)
        trial_ok = False
        for irot in range(Nrot):
            ralpha = np.random.rand() * np.pi * 2;     sinalpha = np.sin(ralpha); cosalpha = np.cos(ralpha)
            rtheta = random_pa() - np.pi / 2; sintheta = np.sin(rtheta); costheta = np.cos(rtheta)
            rphi = np.random.rand() * np.pi * 2;       sinphi = np.sin(rphi);     cosphi = np.cos(rphi)
            rot_r[:, :] = trial.r[1:, :]
            new_x = rot_r[:, 0] * cosalpha - rot_r[:, 1] * sinalpha
            new_y = rot_r[:, 0] * sinalpha + rot_r[:, 1] * cosalpha
            rot_r[:, 0] = new_x; rot_r[:, 1] = new_y
            new_x = rot_r[:, 0] * costheta - rot_r[:, 2] * sintheta
            new_z = rot_r[:, 0] * sintheta + rot_r[:, 2] * costheta
            rot_r[:, 0] = new_x; rot_r[:, 2] = new_z
            new_y = rot_r[:, 1] * cosphi - rot_r[:, 2] * sinphi
            new_z = rot_r[:, 1] * sinphi + rot_r[:, 2] * cosphi
            rot_r[:, 1] = new_y; rot_r[:, 2] = new_z
            for ishift in range(Nshift):
                shift = ((np.random.rand(3) - 0.5) * size).reshape((1, 3))
                shift_r = rot_r + shift
                if room_test_pbc(shift_r, hsigma, DIRECTORY, size):
                    trial_ok = True
                    trial.r[1:, :] = shift_r
                    cells.append(trial)
#                    print('Solvent molecule #%d insertion OK: rotation #%d, shift #%d' % \
#                          (i+1, irot+1, ishift+1))
                    break
            if trial_ok:
                break
        if not trial_ok:
            print('Solvent molecule #%d insertion failed' % (i+1))
            break

    cell = unite(cells, clo, chi)
    return cell


# Forcefield lookup mix-in classes ====================================

class c_ff_pairij_fftypes:

    def FF_PairIJ(self, fftype0, fftype1):
        if fftype0 == fftype1:
            ff_key = '-'.join([fftype0, fftype0])
            try:
                return self.ff_pairij[ff_key]
            except KeyError:
                return []
        try:
            ff_key = '-'.join([fftype0, fftype1])
            return self.ff_pairij[ff_key]
        except KeyError:
            try:
                ff_key = '-'.join([fftype1, fftype0])
                return self.ff_pairij[ff_key]
            except KeyError:
                return []

class c_ff_pairij_subst:

    def FF_PairIJ(self, fftype0, fftype1):
        if (fftype0 not in self.ff_pairij_subst) or (fftype1 not in self.ff_pairij_subst):
            return []

        fftype0 = self.ff_pairij_subst[fftype0]
        fftype1 = self.ff_pairij_subst[fftype1]

        if fftype0 == fftype1:
            ff_key = '-'.join([fftype0, fftype0])
            try:
                return self.ff_pairij[ff_key]
            except KeyError:
                return []
        try:
            ff_key = '-'.join([fftype0, fftype1])
            return self.ff_pairij[ff_key]
        except KeyError:
            try:
                ff_key = '-'.join([fftype1, fftype0])
                return self.ff_pairij[ff_key]
            except KeyError:
                return []

class c_ff_bond_increment_fftypes:

    def FF_Bond_Increment(self, cell, ol0, obond, ol1):     # ol = ordinal number of lateral atom
        'Just by atoms ff types'

        fftype0 = cell.ff[ol0]
        fftype1 = cell.ff[ol1]

        ff_key0 = '-'.join([fftype0, fftype1])
        ff_key1 = '-'.join([fftype1, fftype0])

        if ff_key0 in self.ff_bond_increment:
            return self.ff_bond_increment[ff_key0]
        elif ff_key1 in self.ff_bond_increment:
            return -(self.ff_bond_increment[ff_key1])
        else:
            return None

class c_ff_bond_increment_subst:

    def FF_Bond_Increment(self, cell, ol0, obond, ol1):     # ol = ordinal number of lateral atom
        'Just by atoms ff types'

        fftype0 = cell.ff[ol0]
        fftype1 = cell.ff[ol1]

        if (fftype0 not in self.ff_bond_subst) or (fftype1 not in self.ff_bond_subst):
            return None

        fftype0 = self.ff_bond_increment_subst[fftype0]
        fftype1 = self.ff_bond_increment_subst[fftype1]

        ff_key0 = '-'.join([fftype0, fftype1])
        ff_key1 = '-'.join([fftype1, fftype0])

        if ff_key0 in self.ff_bond_increment:
            return self.ff_bond_increment[ff_key0]
        elif ff_key1 in self.ff_bond_increment:
            return -(self.ff_bond_increment[ff_key1])
        else:
            return None

class c_ff_bond_fftypes:

    def FF_Bond(self, cell, ol0, obond, ol1):     # ol = ordinal number of lateral atom
        'Just by atoms ff types'

        fftype0 = cell.ff[ol0]
        fftype1 = cell.ff[ol1]

        ff_key0 = '-'.join([fftype0, fftype1])
        ff_key1 = '-'.join([fftype1, fftype0])

        if ff_key0 in self.ff_bond:
            return (ff_key0, [ol0, ol1])
        elif ff_key1 in self.ff_bond:
            return (ff_key1, [ol1, ol0])
        else:
            return None

class c_ff_bond_subst:

    def FF_Bond(self, cell, ol0, obond, ol1):     # ol = ordinal number of lateral atom
        'Just by atoms ff types'

        fftype0 = cell.ff[ol0]
        fftype1 = cell.ff[ol1]

        if (fftype0 not in self.ff_bond_subst) or (fftype1 not in self.ff_bond_subst):
            return None

        fftype0 = self.ff_bond_subst[fftype0]
        fftype1 = self.ff_bond_subst[fftype1]

        ff_key0 = '-'.join([fftype0, fftype1])
        ff_key1 = '-'.join([fftype1, fftype0])

        if ff_key0 in self.ff_bond:
            return (ff_key0, [ol0, ol1])
        elif ff_key1 in self.ff_bond:
            return (ff_key1, [ol1, ol0])
        else:
            return None

class c_ff_angle_fftypes:

    def FF_Angle(self, cell, ol0, obondl0c0, oc0, obondl1c0, ol1):     # ol = ordinal number of lateral atom; oc = ... of central atom
        'Just by atoms ff types'

        fftypel0 = cell.ff[ol0]
        fftypec0 = cell.ff[oc0]
        fftypel1 = cell.ff[ol1]

        ff_key0 = '-'.join([fftypel0, fftypec0, fftypel1])
        ff_key1 = '-'.join([fftypel1, fftypec0, fftypel0])

        if ff_key0 in self.ff_angle:
            return [(ff_key0, [ol0, oc0, ol1])]
        elif ff_key1 in self.ff_angle:
            return [(ff_key1, [ol1, oc0, ol0])]
        else:
            return []

class c_ff_angle_subst:

    def FF_Angle(self, cell, ol0, obondl0c0, oc0, obondl1c0, ol1):     # ol = ordinal number of lateral atom; oc = ... of central atom
        'Just by atoms ff types'

        fftypel0 = cell.ff[ol0]
        fftypec0 = cell.ff[oc0]
        fftypel1 = cell.ff[ol1]

        if (fftypel0 not in self.ff_angle_subst) or (fftypel1 not in self.ff_angle_subst)\
        or (fftypec0 not in self.ff_angle_subst):
            return []

        fftypel0 = self.ff_angle_subst[fftypel0]
        fftypec0 = self.ff_angle_subst[fftypec0]
        fftypel1 = self.ff_angle_subst[fftypel1]

        ff_key0 = '-'.join([fftypel0, fftypec0, fftypel1])
        ff_key1 = '-'.join([fftypel1, fftypec0, fftypel0])

        if ff_key0 in self.ff_angle:
            return [(ff_key0, [ol0, oc0, ol1])]
        elif ff_key1 in self.ff_angle:
            return [(ff_key1, [ol1, oc0, ol0])]
        else:
            return []

class c_ff_angle_fftypes_or_subst:

    def FF_Angle(self, cell, ol0, obondl0c0, oc0, obondl1c0, ol1):     # ol = ordinal number of lateral atom; oc = ... of central atom
        'Just by atoms ff types'

        tl0 = cell.ff[ol0]
        tc0 = cell.ff[oc0]
        tl1 = cell.ff[ol1]

        if (tl0 not in self.ff_angle_subst) or (tl1 not in self.ff_angle_subst)\
        or (tc0 not in self.ff_angle_subst):
            return []

        tl0s = self.ff_angle_subst[tl0]
        tc0s = self.ff_angle_subst[tc0]
        tl1s = self.ff_angle_subst[tl1]

        table = [
        ( tl0  +'-'+ tc0  +'-'+ tl1 , [ol0, oc0, ol1]),
        ( tl1  +'-'+ tc0  +'-'+ tl0 , [ol1, oc0, ol0]),
        ( tl0s +'-'+ tc0  +'-'+ tl1 , [ol0, oc0, ol1]),
        ( tl1  +'-'+ tc0  +'-'+ tl0s, [ol1, oc0, ol0]),
        ( tl0  +'-'+ tc0  +'-'+ tl1s, [ol0, oc0, ol1]),
        ( tl1s +'-'+ tc0  +'-'+ tl0 , [ol1, oc0, ol0]),
        ( tl0s +'-'+ tc0  +'-'+ tl1s, [ol0, oc0, ol1]),
        ( tl1s +'-'+ tc0  +'-'+ tl0s, [ol1, oc0, ol0]),
        ( tl0  +'-'+ tc0s +'-'+ tl1 , [ol0, oc0, ol1]),
        ( tl1  +'-'+ tc0s +'-'+ tl0 , [ol1, oc0, ol0]),
        ( tl0s +'-'+ tc0s +'-'+ tl1 , [ol0, oc0, ol1]),
        ( tl1  +'-'+ tc0s +'-'+ tl0s, [ol1, oc0, ol0]),
        ( tl0  +'-'+ tc0s +'-'+ tl1s, [ol0, oc0, ol1]),
        ( tl1s +'-'+ tc0s +'-'+ tl0 , [ol1, oc0, ol0]),
        ( tl0s +'-'+ tc0s +'-'+ tl1s, [ol0, oc0, ol1]),
        ( tl1s +'-'+ tc0s +'-'+ tl0s, [ol1, oc0, ol0]),
        ]

        for ff_key, oatoms in table:
            if ff_key in self.ff_angle:
                return [(ff_key, oatoms)]
        return []

class c_ff_dihedral_fftypes:

    def FF_Dihedral(self, cell, ol0, obondl0c0, oc0, obondc0c1, oc1, obondl1c1, ol1):     # ol = ordinal number of lateral atom; oc = ... of central atom
        'Just by atoms ff types'

        fftypel0 = cell.ff[ol0]
        fftypec0 = cell.ff[oc0]
        fftypec1 = cell.ff[oc1]
        fftypel1 = cell.ff[ol1]

        ff_key0 = '-'.join([fftypel0, fftypec0, fftypec1, fftypel1])
        ff_key1 = '-'.join([fftypel1, fftypec1, fftypec0, fftypel0])

        if ff_key0 in self.ff_dihedral:
            return [(ff_key0, [ol0, oc0, oc1, ol1])]
        elif ff_key1 in self.ff_dihedral:
            return [(ff_key1, [ol1, oc1, oc0, ol0])]
        else:
            return []

class c_ff_dihedral_fftypes_multi:

    def FF_Dihedral(self, cell, ol0, obondl0c0, oc0, obondc0c1, oc1, obondl1c1, ol1):     # ol = ordinal number of lateral atom; oc = ... of central atom
        'Just by atoms ff types. Several entries for each real dihedral are possible'

        fftypel0 = cell.ff[ol0]
        fftypec0 = cell.ff[oc0]
        fftypec1 = cell.ff[oc1]
        fftypel1 = cell.ff[ol1]

        ff_key0 = '-'.join([fftypel0, fftypec0, fftypec1, fftypel1])
        ff_key1 = '-'.join([fftypel1, fftypec1, fftypec0, fftypel0])

        ret = []

        for key in self.ff_dihedral:
            if key[:len(ff_key0)] == ff_key0:
                ret.append( (key, [ol0, oc0, oc1, ol1]) )
            elif key[:len(ff_key1)] == ff_key1:
                ret.append( (key, [ol1, oc1, oc0, ol0]) )

        return ret

class c_ff_dihedral_subst:

    def FF_Dihedral(self, cell, ol0, obondl0c0, oc0, obondc0c1, oc1, obondl1c1, ol1):     # ol = ordinal number of lateral atom; oc = ... of central atom
        'By atoms ff types with substitution table'

        fftypel0 = cell.ff[ol0]
        fftypec0 = cell.ff[oc0]
        fftypec1 = cell.ff[oc1]
        fftypel1 = cell.ff[ol1]

        if (fftypel0 not in self.ff_dihedral_subst) or (fftypel1 not in self.ff_dihedral_subst)\
        or (fftypec0 not in self.ff_dihedral_subst) or (fftypec1 not in self.ff_dihedral_subst):
            return []

        fftypel0 = self.ff_dihedral_subst[fftypel0]
        fftypec0 = self.ff_dihedral_subst[fftypec0]
        fftypec1 = self.ff_dihedral_subst[fftypec1]
        fftypel1 = self.ff_dihedral_subst[fftypel1]

        ff_key0 = '-'.join([fftypel0, fftypec0, fftypec1, fftypel1])
        ff_key1 = '-'.join([fftypel1, fftypec1, fftypec0, fftypel0])

        if ff_key0 in self.ff_dihedral:
            return [(ff_key0, [ol0, oc0, oc1, ol1])]
        elif ff_key1 in self.ff_dihedral:
            return [(ff_key1, [ol1, oc1, oc0, ol0])]
        else:
            return []

class c_ff_dihedral_subst_multi:

    def FF_Dihedral(self, cell, ol0, obondl0c0, oc0, obondc0c1, oc1, obondl1c1, ol1):     # ol = ordinal number of lateral atom; oc = ... of central atom
        'By atoms ff types with substitution table. Several entries for each real dihedral are possible'

        fftypel0 = cell.ff[ol0]
        fftypec0 = cell.ff[oc0]
        fftypec1 = cell.ff[oc1]
        fftypel1 = cell.ff[ol1]

        if (fftypel0 not in self.ff_dihedral_subst) or (fftypel1 not in self.ff_dihedral_subst)\
        or (fftypec0 not in self.ff_dihedral_subst) or (fftypec1 not in self.ff_dihedral_subst):
            return []

        fftypel0 = self.ff_dihedral_subst[fftypel0]
        fftypec0 = self.ff_dihedral_subst[fftypec0]
        fftypec1 = self.ff_dihedral_subst[fftypec1]
        fftypel1 = self.ff_dihedral_subst[fftypel1]

        ff_key0 = '-'.join([fftypel0, fftypec0, fftypec1, fftypel1])
        ff_key1 = '-'.join([fftypel1, fftypec1, fftypec0, fftypel0])

        ret = []

        for key in self.ff_dihedral:
            if key[:len(ff_key0)] == ff_key0:
                ret.append( (key, [ol0, oc0, oc1, ol1]) )
            elif key[:len(ff_key1)] == ff_key1:
                ret.append( (key, [ol1, oc1, oc0, ol0]) )

        return ret


class c_ff_improper_fftypes:

    def FF_Improper(self, cell, oc0, obondl0c0, ol0, obondl1c0, ol1, obondl2c0, ol2):     # ol = ordinal number of lateral atom; oc = ... of central atom
        'Class2 improper by atoms ff types'

        tc0 = cell.ff[oc0]
        tl0 = cell.ff[ol0]
        tl1 = cell.ff[ol1]
        tl2 = cell.ff[ol2]

        table = [
        ( tl0 +'-'+ tc0 +'-'+ tl1 +',-'+ tl2, [ol0, oc0, ol1, ol2]),
        ( tl0 +'-'+ tc0 +'-'+ tl2 +',-'+ tl1, [ol0, oc0, ol2, ol1]),
        ( tl1 +'-'+ tc0 +'-'+ tl0 +',-'+ tl2, [ol1, oc0, ol0, ol2]),
        ( tl2 +'-'+ tc0 +'-'+ tl0 +',-'+ tl1, [ol2, oc0, ol0, ol1]),
        ( tl1 +'-'+ tc0 +'-'+ tl2 +',-'+ tl0, [ol1, oc0, ol2, ol0]),
        ( tl2 +'-'+ tc0 +'-'+ tl1 +',-'+ tl0, [ol2, oc0, ol1, ol0])
        ]

        for ff_key, oatoms in table:
            if ff_key in self.ff_improper:
                return [(ff_key, oatoms)]
        return []

class c_ff_improper_subst:

    def FF_Improper(self, cell, oc0, obondl0c0, ol0, obondl1c0, ol1, obondl2c0, ol2):     # ol = ordinal number of lateral atom; oc = ... of central atom
        'Class2 improper by atoms ff types'

        tc0 = cell.ff[oc0]
        tl0 = cell.ff[ol0]
        tl1 = cell.ff[ol1]
        tl2 = cell.ff[ol2]

        if (tc0 not in self.ff_improper_subst) or (tl0 not in self.ff_improper_subst)\
        or (tl1 not in self.ff_improper_subst) or (tl2 not in self.ff_improper_subst):
            return []

        tc0 = self.ff_improper_subst[tc0]
        tl0 = self.ff_improper_subst[tl0]
        tl1 = self.ff_improper_subst[tl1]
        tl2 = self.ff_improper_subst[tl2]

        table = [
        ( tl0 +'-'+ tc0 +'-'+ tl1 +',-'+ tl2, [ol0, oc0, ol1, ol2]),
        ( tl0 +'-'+ tc0 +'-'+ tl2 +',-'+ tl1, [ol0, oc0, ol2, ol1]),
        ( tl1 +'-'+ tc0 +'-'+ tl0 +',-'+ tl2, [ol1, oc0, ol0, ol2]),
        ( tl2 +'-'+ tc0 +'-'+ tl0 +',-'+ tl1, [ol2, oc0, ol0, ol1]),
        ( tl1 +'-'+ tc0 +'-'+ tl2 +',-'+ tl0, [ol1, oc0, ol2, ol0]),
        ( tl2 +'-'+ tc0 +'-'+ tl1 +',-'+ tl0, [ol2, oc0, ol1, ol0])
        ]

        for ff_key, oatoms in table:
            if ff_key in self.ff_improper:
                return [(ff_key, oatoms)]
        return []

