import src.models.lumped_mas_model as pglmm
import models.lumped_mas_model.llm_models as lmm_models
import matplotlib.pyplot as plt


plt.close("all")

#PG = lmm_models.make_chaari_2006_model()
#PG = lmm_models.make_lin_1999_model()

info_dict = lmm_models.make_chaari_2006_model_w_dict()


PG = pglmm.Planetary_Gear(info_dict)

PG.get_natural_freqs()


