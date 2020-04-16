import src.models.lumped_mas_model as pglmm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import models.lumped_mas_model.llm_models as lmm_models


plt.close("all")

PG = lmm_models.make_chaari_2006_model()
#PG = lmm_models.make_lin_1999_model()

K = PG.K_b + PG.K_e(0) #- (PG.Omega_c)**2 * PG.K_Omega

val, vec = sci.linalg.eig(K, PG.M)
indexes = np.argsort(val)

val = val[indexes]
vec = vec[indexes]

#plt.figure()
#plt.imshow(np.log(vec))
#plt.show()
#np.set_printoptions(threshold=np.inf)


for i in range(len(val)):
    print(np.sqrt(val[i])/(np.pi*2),vec[i])


