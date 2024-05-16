# PCFF parametrization confined to c, h, and f atom types

import simcell as sc

class c_ff(sc.c_ff_bond_increment_fftypes, sc.c_ff_bond_fftypes, sc.c_ff_angle_fftypes, \
sc.c_ff_dihedral_fftypes, sc.c_ff_improper_fftypes):

    def __init__(self, ffparams):
        pass


    ff_mass = {
    "c":   12.01115,
    "h":   1.007970,
    "f":   18.99840,
    }

    def FF_Pair(self, fftype):
        try:
            return self.ff_pair[fftype]
        except:
            return []

    ff_pair_style = 'lj/class2'

    ff_pair = {
    "c":  [0.05400, 4.0100],
    "h":  [0.02000, 2.9950],
    "f":  [0.05980, 3.2000],
    }

    ff_bond_increment = {
    "c-c"  :   0.0000,
    "c-h"  :  -0.0530,
    "c-f"  :   0.2500,
    }

    ff_bond_style = 'class2'

    ff_bond_subst = {
    "Ch":  "c",
    "Cf":  "c",
    "Cx":  "c",
    "H_":  "h",
    "F_":  "f",
    "Cl":  "cl",
    }

    ff_bond = {
    "c-c"  :   {'Bond_Coeffs': [1.5300, 299.6700, -501.7700,  679.8100], 'Order': 1},
    "c-h"  :   {'Bond_Coeffs': [1.1010, 345.0000, -691.8900,  844.6000], 'Order': 1},
    "c-f"  :   {'Bond_Coeffs': [1.3900, 403.0320,    0.0000,    0.0000], 'Order': 1},
    }

    ff_angle_style = 'class2'

    ff_angle = {
    "c-c-c" :    {'Angle_Coeffs': [112.6700, 39.5160,  -7.4430, -9.5583], 'BondBond_Coeffs': [ 0.0000, 1.5300, 1.5300], 'BondAngle_Coeffs': [ 8.0160,  8.0160, 1.5300, 1.5300]},
    "c-c-h" :    {'Angle_Coeffs': [110.7700, 41.4530, -10.6040,  5.1290], 'BondBond_Coeffs': [ 3.3872, 1.5300, 1.1010], 'BondAngle_Coeffs': [20.7540, 11.4210, 1.5300, 1.1010]},
    "h-c-h" :    {'Angle_Coeffs': [107.6600, 39.6410, -12.9210, -2.4318], 'BondBond_Coeffs': [ 5.3316, 1.1010, 1.1010], 'BondAngle_Coeffs': [18.1030, 18.1030, 1.1010, 1.1010]},
    "c-c-f" :    {'Angle_Coeffs': [109.2000, 68.3715,   0.0000,  0.0000], 'BondBond_Coeffs': [ 0.0000, 1.5300, 1.3900], 'BondAngle_Coeffs': [ 0.0000,  0.0000, 1.5300, 1.3900]},
    "f-c-f" :    {'Angle_Coeffs': [109.1026, 71.9700,   0.0000,  0.0000], 'BondBond_Coeffs': [ 0.0000, 1.3900, 1.3900], 'BondAngle_Coeffs': [ 0.0000,  0.0000, 1.3900, 1.3900]},
    "f-c-h" :    {'Angle_Coeffs': [108.5010, 57.5760,   0.0000,  0.0000], 'BondBond_Coeffs': [ 0.0000, 1.3900, 1.1010], 'BondAngle_Coeffs': [ 0.0000,  0.0000, 1.3900, 1.1010]},
    }

    ff_dihedral_style = 'class2'

    ff_dihedral = {
    "c-c-c-c" :     {'Dihedral_Coeffs': [ 0.0000, 0.0,  0.0514, 0.0, -0.1430, 0.0], 'MiddleBondTorsion_Coeffs': [-17.7870, -7.1877,  0.0000, 1.5300], 'EndBondTorsion_Coeffs': [-0.0732,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 1.5300, 1.5300], 'AngleTorsion_Coeffs': [ 0.3886, -0.3139,  0.1389,  0.3886, -0.3139,  0.1389, 112.6700, 112.6700], 'AngleAngleTorsion_Coeffs': [-22.0450, 112.6700, 112.6700], 'BondBond13_Coeffs': [ 0.0000, 1.5300, 1.5300]},
    "c-c-c-h" :     {'Dihedral_Coeffs': [ 0.0000, 0.0,  0.0316, 0.0, -0.1681, 0.0], 'MiddleBondTorsion_Coeffs': [-14.8790, -3.6581, -0.3138, 1.5300], 'EndBondTorsion_Coeffs': [ 0.2486,  0.2422, -0.0925,  0.0814,  0.0591,  0.2219, 1.5300, 1.1010], 'AngleTorsion_Coeffs': [-0.2454,  0.0000, -0.1136,  0.3113,  0.4516, -0.1988, 112.6700, 110.7700], 'AngleAngleTorsion_Coeffs': [-16.1640, 112.6700, 110.7700], 'BondBond13_Coeffs': [ 0.0000, 1.5300, 1.1010]},
    "h-c-c-h" :     {'Dihedral_Coeffs': [-0.1432, 0.0,  0.0617, 0.0, -0.1530, 0.0], 'MiddleBondTorsion_Coeffs': [-14.2610, -0.5322, -0.4864, 1.5300], 'EndBondTorsion_Coeffs': [ 0.2130,  0.3120,  0.0777,  0.2130,  0.3120,  0.0777, 1.1010, 1.1010], 'AngleTorsion_Coeffs': [-0.8085,  0.5569, -0.2466, -0.8085,  0.5569, -0.2466, 110.7700, 110.7700], 'AngleAngleTorsion_Coeffs': [-12.5640, 110.7700, 110.7700], 'BondBond13_Coeffs': [ 0.0000, 1.1010, 1.1010]},
    "c-c-c-f" :     {'Dihedral_Coeffs': [ 0.0000, 0.0,  0.0000, 0.0,  0.1500, 0.0], 'MiddleBondTorsion_Coeffs': [  0.0000,  0.0000,  0.0000, 1.5300], 'EndBondTorsion_Coeffs': [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 1.1010, 1.1010], 'AngleTorsion_Coeffs': [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 110.7700, 110.7700], 'AngleAngleTorsion_Coeffs': [  0.0000, 110.7700, 110.7700], 'BondBond13_Coeffs': [ 0.0000, 1.1010, 1.1010]},
    "f-c-c-f" :     {'Dihedral_Coeffs': [ 0.0000, 0.0,  0.0000, 0.0, -0.1000, 0.0], 'MiddleBondTorsion_Coeffs': [  0.0000,  0.0000,  0.0000, 1.5300], 'EndBondTorsion_Coeffs': [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 1.1010, 1.1010], 'AngleTorsion_Coeffs': [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 110.7700, 110.7700], 'AngleAngleTorsion_Coeffs': [  0.0000, 110.7700, 110.7700], 'BondBond13_Coeffs': [ 0.0000, 1.1010, 1.1010]},
    "f-c-c-h" :     {'Dihedral_Coeffs': [ 0.0000, 0.0,  0.0000, 0.0, -0.1000, 0.0], 'MiddleBondTorsion_Coeffs': [  0.0000,  0.0000,  0.0000, 1.5300], 'EndBondTorsion_Coeffs': [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 1.1010, 1.1010], 'AngleTorsion_Coeffs': [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 110.7700, 110.7700], 'AngleAngleTorsion_Coeffs': [  0.0000, 110.7700, 110.7700], 'BondBond13_Coeffs': [ 0.0000, 1.1010, 1.1010]},
    }

    ff_improper_style = 'class2'

    ff_improper = {
    "c-c-c,-h":     {'Improper_Coeffs': [0.0000, 0.0000], 'AngleAngle_Coeffs': [-1.3199, -1.3199,  0.1184, 112.6700, 110.7700, 110.7700]},
    "h-c-c,-h":     {'Improper_Coeffs': [0.0000, 0.0000], 'AngleAngle_Coeffs': [-0.4825,  0.2738,  0.2738, 110.7700, 107.6600, 110.7700]},
    "c-c-h,-c":     {'Improper_Coeffs': [0.0000, 0.0000], 'AngleAngle_Coeffs': [ 0.1184, -1.3199, -1.3199, 110.7700, 112.6700, 110.7700]},
    "c-c-h,-h":     {'Improper_Coeffs': [0.0000, 0.0000], 'AngleAngle_Coeffs': [ 0.2738, -0.4825,  0.2738, 110.7700, 110.7700, 107.6600]},
    }


