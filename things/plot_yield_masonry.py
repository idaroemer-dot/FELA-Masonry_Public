import numpy as np
import matplotlib.pyplot as plt

from fem.constrains_masonry import Ab_masonry_ps

# --- material ---
ftx = 300e6
fcm = 20e6
nu  = 0.2
fcl = 2e6
phi_l_deg = 30
n_ix = 4

# hent constraints
Aseq, Aaeq, byeq, As, Aa, by = Ab_masonry_ps(
    ftx, fcm, nu, fcl, phi_l_deg, n_ix=n_ix
)

# vi kigger kun på inequalities (Aa)
# variabler = [sx, sy, tau, ...]
# vi sætter sx = 0

sy_vals = np.linspace(-fcl*1.5, 0, 200)
tau_max = []

for sy in sy_vals:
    tau_candidates = []

    for i in range(Aa.shape[0]):
        a = Aa[i]
        b = by[i]

        # a = [sx, sy, tau, ...]
        a_sy = a[1]
        a_tau = a[6]   # <- vigtigt: tau index i din model

        # constraint: a_sy * sy + a_tau * tau ≤ b

        if abs(a_tau) > 1e-12:
            tau_bound = (b - a_sy * sy) / a_tau
            tau_candidates.append(tau_bound)

    if len(tau_candidates) > 0:
        tau_max.append(min(tau_candidates))
    else:
        tau_max.append(0)

tau_max = np.array(tau_max)

# --- plot ---
plt.figure(figsize=(6,5))
plt.plot(sy_vals/1e6, tau_max/1e6, label="upper")
plt.plot(sy_vals/1e6, -tau_max/1e6, label="lower")

plt.xlabel(r"$\sigma_y$ [MPa]")
plt.ylabel(r"$\tau_{xy}$ [MPa]")
plt.title("Masonry yield surface (σx = 0)")
plt.grid(True)
plt.legend()
plt.show()