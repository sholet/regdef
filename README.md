An utility for construction of atomistic systems studied in the paper        
Modeling impact of regiodefects on electrocaloric effect in poly(VDF-co-TrFE) copolymer        
by Vadim I. Sultanov, Vadim V. Atrazhev, and Dmitry V. Dmitriev        
to be published in The Journal of Physical Chemistry B


Prerequisites: Python installation, NumPy module


`con_copol_regdef.py` script produces a LAMMPS data file, a text map file with monomer unit sequences are listed, and car + mdf file pair for system visualization in Biovia Materials Studio Visualizer


To obtain three systems studied in the paper one should execute these three commands:        
```
con_copol_regdef.py 0pc.data defectrate=0.00 mx=4 my=8 mz=50 a=8.8 b=4.85 content="vdf 1760 trfe-S 720 trfe-R 720"
con_copol_regdef.py 3pc.data defectrate=0.03 mx=4 my=8 mz=50 a=8.8 b=4.85 content="vdf 1760 trfe-S 720 trfe-R 720"
con_copol_regdef.py 6pc.data defectrate=0.06 mx=4 my=8 mz=50 a=8.8 b=4.85 content="vdf 1760 trfe-S 720 trfe-R 720"
```


Note that since a random number generator is involved the obtained systems will not be quite
the same as those studied by the authors. However, the obtained systems should demonstrate
the same behavior in molecular dynamics simulations.
