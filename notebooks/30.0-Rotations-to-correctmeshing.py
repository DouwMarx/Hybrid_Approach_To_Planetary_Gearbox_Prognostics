import src.models.lumped_mas_model as pglmm
import models.lumped_mas_model.llm_models as lmm_models
import matplotlib.pyplot as plt
import numpy as np


plt.close("all")


#info_dict = lmm_models.make_chaari_2006_model_w_dict()
info_dict = lmm_models.make_bonfiglioli()

PG = pglmm.Planetary_Gear(info_dict)

r_obj = pglmm.PG_ratios(PG.Z_r,PG.Z_s,PG.Z_p)

