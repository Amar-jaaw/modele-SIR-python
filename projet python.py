import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# MODELE SIR
def sir_model(state, t, B, A, N):
    S, I, R = state
    dSdt = -B * S * I / N
    dIdt = B * S * I / N - A * I
    dRdt = A * I
    return dSdt, dIdt, dRdt


# PARAMETRES
B = 0.3   # taux de transmission
A = 0.1   # taux de guérison
N = 10000 # population totale

I0 = 1    # infectés initiaux
R0 = 0    # guéris initiaux
S0 = N - I0 - R0

days = 100


# TEMPS
t = np.linspace(0, days, 1000)

# RESOLUTION
solution = odeint(sir_model, [S0, I0, R0], t, args=(B, A, N))
S, I, R = solution.T


# RESULTATS
peak = int(max(I))
peak_day = t[np.argmax(I)]

print("Pic d'infectés :", peak)
print("Jour du pic :", round(peak_day, 1))


# GRAPHIQUE
plt.plot(t, S, label="Susceptibles")
plt.plot(t, I, label="Infectés")
plt.plot(t, R, label="Guéris")

plt.title("Modèle SIR")
plt.xlabel("Temps (jours)")
plt.ylabel("Population")
plt.legend()
plt.grid()

plt.show()


