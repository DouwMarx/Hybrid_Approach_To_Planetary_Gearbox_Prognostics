
import models.lumped_mas_model.llm_models as lmm
import numpy as np
import matplotlib.pyplot as plt
import definitions
import src.models.lumped_mas_model as pglmm
import dill

plt.close("all")



def save_model():
    with open(definitions.root + "\\models\\solved_models" + "\\" + "get_y" + ".lmmsol", 'wb') as config:
        PG = lmm.make_chaari_2006_model()
        PG.get_solution()
        dill.dump(PG, config)
    return


def load_model():
        with open(definitions.root + "\\models\\solved_models" + "\\" + "sol_save_test_w_sol" + ".lmmsol", 'rb') as config:
            PG = dill.load(config)
        return PG

PG = load_model()
#PG.plot_solution("Displacement")

transp = pglmm.Transmission_Path(PG)
y = transp.y()
plt.figure()
plt.plot(y)
#plt.plot(transp.F_ri(1, t))


