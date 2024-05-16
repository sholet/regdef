#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 22:53:11 2021

@author: Vadim Sultanov

Polymer crystal with finite chains containing regiodefects
construction utility

For each monomer with name 'mononame' two monomer unit definition
.data files must exist:
1) mononame-for.data
2) mononame-bac.data
Requirements to the monomer unit definition .data files:
1) the first backbone atom must have number 1
2) the last bond must link the last backbone atom to atom N+1,
   where N is a number of atoms in this monomer unit
   (i.e. to the first backbone atom of the next monomer unit)
Bond types do not matter; they will be replaced by the
correct ones on application of the force field.
All the monomer units must have the same size in z direction.
Their sizes in x and y directions do not matter.
"""

import numpy as np
import sys
import os.path as osp
from collections import Counter
import simcell as sc
import random
from copy import deepcopy
from datetime import datetime

# Command line options #########################################################
outname = 'copolymer.data'     # name of output file
mapname = ''                   # name of map file (scheme of monomers
                               # location along chains)
ff = 'ff_pcff_chf'             # force field to apply
term_type = 3                  # type of terminal atoms according to .data files
term_inc = 0.25                # charge: C-terminal atom bond increment
a = 0.0                        # elementary cell a and b sizes
b = 0.0                        # MUST be specified
ending = 3.0                   # addition to box z-size to accommodate terminal atoms
mx = 4                         # number of elementary cells in x, y, and z
my = 8                         # dimensions respectively; ignored if a map
mz = 50                        # file is provided
defectrate = 0.06              # the next monomer adds in an opposite direction
                               # with this probability;
                               # ignored if a map file is provided
content = ''                   # monomer names interleaved with their numbers
                               # e.g. 'vdf 526 trfe-S 103 trfe-R 103'
                               # ignored if a map file is provided
################################################################################

x, y, z = 0, 1, 2

def getkind(urn):
    Nkind = len(urn)
    r = random.random() * sum(urn)
    uplim = 0
    for i in range(Nkind):
        uplim += urn[i]
        if r <= uplim:
            urn[i] -= 1
            return i

################################################################################

import ast
namedargs = [arg for arg in sys.argv if '=' in arg]
positargs = [arg for arg in sys.argv if '=' not in arg][1:]
for arg in namedargs:
    arg_name, raw_value = arg.split('=')[:2]
    if raw_value == '': raw_value = 'True'
    try: arg_value = ast.literal_eval(raw_value)
    except: arg_value = raw_value
    globals()[arg_name] = arg_value

if len(positargs) > 0:
    outname = positargs[0]
fname, fext = osp.splitext(outname)

hmz = mz // 2

maplist = []
displist = []
comment = ''
if mapname != '':
    with open(mapname, 'r') as mapfile:
        ll = mapfile.readlines()
    maplines = deepcopy(ll)
    measures = ll.pop(0)
    exec(measures)
    content = ll.pop(0)
    content_s = content.split()
    mononames = content_s[::2]
    cell = sc.c_cell(mononames[0] + '-for.data')
    c = cell.zhi - cell.zlo
    rateline = ll.pop(0)
    defectrate = float(rateline.split()[-1])
    displacements = ll.pop(0)
    displacements = displacements.split()
    displist = [int(d) for d in displacements[1:]]
    ll.pop(0)
    ls = [l.split() for l in ll]
    Nkinds = len(mononames)
    ml = [l[1:] for l in ls[len(mononames)*2:]]
    ml = zip(*ml)
    for line in ml:
        chkinds = []
        for pair in line:
            kind, dirn = pair.split('|')
            kind = int(kind)
            dirn = bool(int(dirn))
            chkinds.append( (kind, dirn) )
        maplist.append(chkinds)
else:
    content_s = content.split()
    mononames = content_s[::2]
    urn = [int(s) for s in content_s[1::2]]
    cell = sc.c_cell(mononames[0] + '-for.data')
    c = cell.zhi - cell.zlo
    Nkinds = len(mononames)
    if Nkinds != len(urn):
        print(f"Warning: bad 'content' parameter")
        
    sb = sum(urn); dimprod = 2 * mx * my * mz
    if sb != dimprod:
        print(f"Warning: urn size is not equal to twice number of elementary cells ({sb} != {dimprod})")

    reversions = []
    for ichain in range(mx * my * 2):
        displist.append(random.randint(-hmz, hmz))
        chkinds = []
        Ndef = 0     # number of regiodefects
        Nbac = 0     # number of monomer units with backward direction
        dirn = False # monomer unit direction: forward (False) or backward (True)
        for imon in range(mz):
            kind = getkind(urn)
            if random.random() < defectrate and imon != 0:
                dirn = not dirn
                Ndef += 1
            Nbac += int(dirn)
            chkinds.append( (kind, dirn) )
        if Nbac > mz // 2:
            chkinds = [(kind, not dirn) for kind, dirn in chkinds]
        maplist.append(chkinds)
        reversions.append(Ndef)
    counts = [Counter(chkinds) for chkinds in maplist]
    mapname = f'{fname}.map.txt'
    maplines = []
    maplines.append(f'mx = {mx}; my = {my}; mz = {mz}\n')
    maplines.append(content + '\n')
    maplines.append(f'Regiodefect rate {defectrate}\n')
    mapline = '\t'.join([str(disp) for disp in displist])
    maplines.append('displacement:\t' + mapline + '\n')
    mapline = '\t'.join([str(Nrev) for Nrev in reversions])
    maplines.append('regiodefects:\t' + mapline + '\n')
    for kind in range(len(mononames)):
        for dirn in (False, True):
            mapline = '\t'.join([str(ct[(kind, dirn)]) for ct in counts])
            maplines.append(f'{mononames[kind]} ({kind}|{int(dirn)}):\t' + mapline + '\n')
    for imon in range(mz):
        mapline = '\t'.join([f'{chkinds[imon][0]}|{int(chkinds[imon][1])}' for chkinds in maplist])
        maplines.append(f'{imon}\t' + mapline + '\n')
    with open(mapname, 'w') as mapfile:
        for mapline in maplines:
            mapfile.write(mapline)

commentlines = [outname + '\n']
commentlines.append('Constructed by con_copol_regdef.py at ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S\n'))
commentlines.append(f'Force field: {ff}\n')
commentlines.append(f'term_type = {term_type}; term_inc = {term_inc}; ending = {ending}\n')
commentlines.append(f'a = {a}; b = {b}; c = {c}\n')
commentlines.append(f'Map file: {mapname}\n')
# commentlines += maplines
cell.comment = ''.join(commentlines)

formonomers = [sc.c_cell(mn + '-for.data') for mn in mononames]    # forward monomers
bacmonomers = [sc.c_cell(mn + '-bac.data') for mn in mononames]    # backward monomers
monomers = [formonomers, bacmonomers]

Namon = [mon.N for mon in formonomers]
Nbmon = [mon.Nbonds for mon in formonomers]
Nachain = [sum([Namon[kind] for kind, dirn in chkinds]) + 2 for chkinds in maplist]
Nbchain = [sum([Nbmon[kind] for kind, dirn in chkinds]) + 1 for chkinds in maplist]
cell.N = sum(Nachain)
cell.Nbonds = sum(Nbchain)
cell.Nmols = len(maplist)

ha = a / 2                    # half a
hb = b / 2                    # half b
hc = c / 2                    # half c
hxsize = mx * ha              # half x, y and z size of the final cell
hysize = my * hb
hzsize = mz * hc
cell.xlo = -hxsize
cell.xhi =  hxsize
cell.ylo = -hysize
cell.yhi =  hysize
cell.zlo = -hzsize - ending
cell.zhi =  hzsize + ending

cell.omol = np.zeros(cell.N + 1, dtype = int)
cell.q    = np.zeros(cell.N + 1)
cell.r    = np.zeros((cell.N + 1, 3))
cell.t    = np.zeros(cell.N + 1, dtype = np.uint8)
cell.text = np.zeros(cell.N + 1, dtype = object)

cell.bond = np.zeros((cell.Nbonds + 2, 2), dtype = int)
cell.bond_t = np.zeros(cell.Nbonds + 2, dtype = np.uint8)
cell.bond_order = np.zeros(cell.Nbonds + 2, dtype = np.uint8)

cell.mol_name = np.zeros(cell.Nmols + 1, dtype = object)

cell.mol_1st  = np.zeros(cell.Nmols + 2, dtype = int)
cell.mol_b1st = np.zeros(cell.Nmols + 2, dtype = int)

def seqsum(seq):
    ret = [0]
    for n in seq:
        ret.append(n + ret[-1])
    return ret[1:]

cell.mol_1st[1] = 1
cell.mol_1st[2:] = seqsum(Nachain)
cell.mol_1st[2:] += 1
cell.mol_b1st[1] = 1
cell.mol_b1st[2:] = seqsum(Nbchain)
cell.mol_b1st[2:] += 1

for ich in range(1, cell.Nmols + 1):
    cell.omol[cell.mol_1st[ich]: cell.mol_1st[ich+1]] = ich

formono_r = [[], []]
bacmono_r = [[], []]
for iside in range(2):
    devsign = (-1)**iside
    for imon in range(Nkinds):
        mon = formonomers[imon]
        r = np.zeros((mon.N, 3))
        r[:, :] = mon.r[1:, :]
        r[:, x] += devsign * 0.08 * r[:, y] * (r[:, y] > 0.0)
        formono_r[iside].append(r)
        
        mon = bacmonomers[imon]
        r = np.zeros((mon.N, 3))
        r[:, :] = mon.r[1:, :]
        r[:, x] -= devsign * 0.08 * r[:, y] * (r[:, y] < 0.0)
        bacmono_r[iside].append(r)
mono_r = [formono_r, bacmono_r]

xbase = 0.0; ybase = 0.0
oatombase = 1
obondbase = 1
omol = 0
for icol in range(mx):
    for irow in range(2*my):
        zbase = c * displist[icol*my*2 + irow]
        chainobase = oatombase
        omol += 1
        cell.mol_name[omol] = f'{icol:02d}_{irow:02d}'
        for imon in range(mz):
            kind, dirn = maplist[icol*my*2 + irow][imon]
            cell.r[oatombase: oatombase + monomers[dirn][kind].N] = \
                mono_r[dirn][imon%2][kind] + np.array([[xbase, ybase, zbase],])
            cell.q[oatombase: oatombase + monomers[dirn][kind].N] = \
                monomers[dirn][kind].q[1:]
            cell.t[oatombase: oatombase + monomers[dirn][kind].N] = \
                monomers[dirn][kind].t[1:]
            cell.text[oatombase: oatombase + monomers[dirn][kind].N] = \
                monomers[dirn][kind].text[1:]
            cell.bond[obondbase: obondbase + monomers[dirn][kind].Nbonds] = \
                monomers[dirn][kind].bond[1:] + (oatombase - 1)
            cell.bond_t[obondbase: obondbase + monomers[dirn][kind].Nbonds] = \
                monomers[dirn][kind].bond_t[1:]
            cell.bond_order[obondbase: obondbase + monomers[dirn][kind].Nbonds] = \
                monomers[dirn][kind].bond_order[1:]
            oatombase += monomers[dirn][kind].N
            obondbase += monomers[dirn][kind].Nbonds
            zbase += c
        # terminal F at chain end
        chainolast = cell.bond[obondbase-1][0]
        cell.r[oatombase] = cell.r[chainolast] + (0.0, -0.805201, 1.133028)
        cell.q[oatombase] = -term_inc
        cell.q[chainolast] += term_inc
        cell.t[oatombase] = term_type
        oatombase += 1
        # terminal F at chain beginning
        cell.r[oatombase] = cell.r[chainobase] + (0.0, 0.805201, -1.133028)
        cell.q[oatombase] = -term_inc
        cell.q[chainobase] += term_inc
        cell.t[oatombase] = term_type
        cell.bond[obondbase, :] = (oatombase, chainobase)
        cell.bond_t[obondbase] = 1
        cell.bond_order[obondbase] = 1
        obondbase += 1
        oatombase += 1
        
        xbase += ha if irow % 2 == 0 else -ha
        ybase += hb
    xbase += a
    ybase = 0.0

cell.bond = np.delete(cell.bond, -1, 0)
cell.bond_t = np.delete(cell.bond_t, -1, 0)
cell.bond_order = np.delete(cell.bond_order, -1, 0)

cell.apply_ff(ff)

cellcom = cell.com()
cell.r[1:, :] -= cellcom

cell.write_data(f'{fname}.data')
# files for visualization in Biovia Materials Studio Visualizer
cell.write_mdf(f'{fname}.mdf')
cell.write_car(f'{fname}.car')

