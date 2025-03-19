import numpy as np 
import matplotlib.pyplot as plt
from EconModel import jit
from HeterogenousFirmsModel import HeterogenousFirmsModelClass

model = HeterogenousFirmsModelClass(name = 'HeterogenousFirmsModel')

model.prepare()
model.solve_steady_state()

with jit(model) as model:
    par = model.par
    ss = model.ss 

    