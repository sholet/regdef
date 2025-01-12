#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 

@author: hoshea
"""

class c_element:
    def __init__(self, o, symbol, name, m):
        self.o = o
        self.symbol = symbol
        self.name = name
        self.m = m

elements = [
c_element(1,   'H' , 'Hydrogen',      1.007941),
c_element(2,   'He', 'Helium',        4.0026022),
c_element(3,   'Li', 'Lithium',       6.9412),
c_element(4,   'Be', 'Beryllium',     9.0121822),
c_element(5,   'B' , 'Boron',         10.8112),
c_element(6,   'C' , 'Carbon',        12.01072),
c_element(7,   'N' , 'Nitrogen',      14.00672),
c_element(8,   'O' , 'Oxygen',        15.99942),
c_element(9,   'F' , 'Fluorine',      18.99840322),
c_element(10,  'Ne', 'Neon',          20.17972),
c_element(11,  'Na', 'Sodium',        22.989769282),
c_element(12,  'Mg', 'Magnesium',     24.3052),
c_element(13,  'Al', 'Aluminium',     26.98153862),
c_element(14,  'Si', 'Silicon',       28.08552),
c_element(15,  'P' , 'Phosphorus',    30.9737622),
c_element(16,  'S' , 'Sulfur',        32.0652),
c_element(17,  'Cl', 'Chlorine',      35.4532),
c_element(18,  'Ar', 'Argon',         39.9482),
c_element(19,  'K' , 'Potassium',     39.09832),
c_element(20,  'Ca', 'Calcium',       40.0782),
c_element(21,  'Sc', 'Scandium',      44.9559122),
c_element(22,  'Ti', 'Titanium',      47.8672),
c_element(23,  'V' , 'Vanadium',      50.94152),
c_element(24,  'Cr', 'Chromium',      51.99612),
c_element(25,  'Mn', 'Manganese',     54.9380452),
c_element(26,  'Fe', 'Iron',          55.8452),
c_element(27,  'Co', 'Cobalt',        58.9331952),
c_element(28,  'Ni', 'Nickel',        58.69342),
c_element(29,  'Cu', 'Copper',        63.5462),
c_element(30,  'Zn', 'Zinc',          65.382),
c_element(31,  'Ga', 'Gallium',       69.7232),
c_element(32,  'Ge', 'Germanium',     72.632),
c_element(33,  'As', 'Arsenic',       74.92162),
c_element(34,  'Se', 'Selenium',      78.962),
c_element(35,  'Br', 'Bromine',       79.9042),
c_element(36,  'Kr', 'Krypton',       83.7982),
c_element(37,  'Rb', 'Rubidium',      85.46782),
c_element(38,  'Sr', 'Strontium',     87.622),
c_element(39,  'Y' , 'Yttrium',       88.905852),
c_element(40,  'Zr', 'Zirconium',     91.2242),
c_element(41,  'Nb', 'Niobium',       92.906382),
c_element(42,  'Mo', 'Molybdenum',    95.962),
c_element(43,  'Tc', 'Technetium',    98.0),
c_element(44,  'Ru', 'Ruthenium',     101.072),
c_element(45,  'Rh', 'Rhodium',       102.90552),
c_element(46,  'Pd', 'Palladium',     106.422),
c_element(47,  'Ag', 'Silver',        107.86822),
c_element(48,  'Cd', 'Cadmium',       112.4112),
c_element(49,  'In', 'Indium',        114.8182),
c_element(50,  'Sn', 'Tin',           118.712),
c_element(51,  'Sb', 'Antimony',      121.762),
c_element(52,  'Te', 'Tellurium',     127.62),
c_element(53,  'I' , 'Iodine',        126.904472),
c_element(54,  'Xe', 'Xenon',         131.2932),
c_element(55,  'Cs', 'Caesium',       132.90545192),
c_element(56,  'Ba', 'Barium',        137.3272),
c_element(57,  'La', 'Lanthanum',     138.905472),
c_element(58,  'Ce', 'Cerium',        140.1162),
c_element(59,  'Pr', 'Praseodymium',  140.907652),
c_element(60,  'Nd', 'Neodymium',     144.2422),
c_element(61,  'Pm', 'Promethium',    145.0),
c_element(62,  'Sm', 'Samarium',      150.362),
c_element(63,  'Eu', 'Europium',      151.9642),
c_element(64,  'Gd', 'Gadolinium',    157.252),
c_element(65,  'Tb', 'Terbium',       158.925352),
c_element(66,  'Dy', 'Dysprosium',    162.52),
c_element(67,  'Ho', 'Holmium',       164.930322),
c_element(68,  'Er', 'Erbium',        167.2592),
c_element(69,  'Tm', 'Thulium',       168.934212),
c_element(70,  'Yb', 'Ytterbium',     173.0542),
c_element(71,  'Lu', 'Lutetium',      174.96682),
c_element(72,  'Hf', 'Hafnium',       178.492),
c_element(73,  'Ta', 'Tantalum',      180.947882),
c_element(74,  'W' , 'Tungsten',      183.842),
c_element(75,  'Re', 'Rhenium',       186.2072),
c_element(76,  'Os', 'Osmium',        190.232),
c_element(77,  'Ir', 'Iridium',       192.2172),
c_element(78,  'Pt', 'Platinum',      195.0842),
c_element(79,  'Au', 'Gold',          196.9665692),
c_element(80,  'Hg', 'Mercury',       200.592),
c_element(81,  'Tl', 'Thallium',      204.38332),
c_element(82,  'Pb', 'Lead',          207.22),
c_element(83,  'Bi', 'Bismuth',       208.98042),
c_element(84,  'Po', 'Polonium',      209.0),
c_element(85,  'At', 'Astatine',      210.0),
c_element(86,  'Rn', 'Radon',         222.0),
c_element(87,  'Fr', 'Francium',      223.0),
c_element(88,  'Ra', 'Radium',        226.0),
c_element(89,  'Ac', 'Actinium',      227.0),
c_element(90,  'Th', 'Thorium',       232.038062),
c_element(91,  'Pa', 'Protactinium',  231.035882),
c_element(92,  'U' , 'Uranium',       238.028912),
c_element(93,  'Np', 'Neptunium',     237.0),
c_element(94,  'Pu', 'Plutonium',     244.0),
c_element(95,  'Am', 'Americium',     243.0),
c_element(96,  'Cm', 'Curium',        247.0),
c_element(97,  'Bk', 'Berkelium',     247.0),
c_element(98,  'Cf', 'Californium',   251.0),
c_element(99,  'Es', 'Einsteinium',   252.0),
c_element(100, 'Fm', 'Fermium',       257.0),
c_element(101, 'Md', 'Mendelevium',   258.0),
c_element(102, 'No', 'Nobelium',      259.0),
c_element(103, 'Lr', 'Lawrencium',    262.0),
c_element(104, 'Rf', 'Rutherfordium', 267.0),
c_element(105, 'Db', 'Dubnium',       268.0),
c_element(106, 'Sg', 'Seaborgium',    271.0),
c_element(107, 'Bh', 'Bohrium',       272.0),
c_element(108, 'Hs', 'Hassium',       270.0),
c_element(109, 'Mt', 'Meitnerium',    276.0),
c_element(110, 'Ds', 'Darmstadtium',  281.0),
c_element(111, 'Rg', 'Roentgenium',   280.0),
c_element(112, 'Cn', 'Copernicium',   285.0)
]

No = {}
Ar = {}
for elm in elements:
    No[elm.symbol] = elm.o
    Ar[elm.symbol] = elm.m
    Ar[elm.o] = elm.m
