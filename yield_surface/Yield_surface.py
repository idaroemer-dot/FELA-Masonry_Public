import numpy as np
import matplotlib.pyplot as plt

# material parameters
params = {
    # -------- basic strengths --------
    "ftx": 0.32,                  # MPa
    "fcm": 7.56,                  # MPa, used in surface II
    "nu": 0.6,                      # VIII

    # -------- surface X --------
    "fcs": 15.41,                 # MPa
    "fce": 2.40,                  # MPa
    "fts": 15.41 / 20.0,          # MPa
    "phi_s": np.deg2rad(30.0),    # rad
    "xi": 0.5,                    # As / Atot

    # -------- surface XI --------
    "fcl": 2.40,                  # MPa
    "phi_l": np.deg2rad(30.0),    # rad
    "omega_max": np.deg2rad(24.0) # rad
}

# Pages data for plotting
from page_data import page_data


# transformation
#    Fra principal stresses (sigma1, sigma2) til
#    materialekoordinater (sigma_x, sigma_y, tau_xy)
#    theta = vinkel mellem hovedspændingsretning og materialeakser

def principal_to_material(sig1, sig2, theta_deg):
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)

    sigma_x = sig1 * c**2 + sig2 * s**2
    sigma_y = sig1 * s**2 + sig2 * c**2
    tau_xy  = (sig1 - sig2) * s * c

    return sigma_x, sigma_y, tau_xy


# min flydefalde

def compute_ls_ms(p):
    fcs = p["fcs"]
    fts = p["fts"]
    phi_s = p["phi_s"]

    sin_phi_s = np.sin(phi_s)

    ls = 1.0 - 2.0 * (fts / fcs) * (sin_phi_s / (1.0 - sin_phi_s))
    ms = 1.0 - 2.0 * (fts / fcs) * (1.0 / (1.0 - sin_phi_s))

    return ls, ms

def tau_X(sx, p):
    fcs = p["fcs"]
    fce = p["fce"]
    xi  = p["xi"]

    ls, ms = compute_ls_ms(p)

    A = 0.5 * fcs * xi * (ls - ms) - sx
    B = 0.5 * fcs * xi * (ls + ms) + fce * (1.0 - xi) + sx

    value = A * B

    if value <= 0.0:
        return 0.0

    return np.sqrt(value)

def tau_XI(sx, sy, p):
    fcl = p["fcl"]
    fce = p["fce"]
    phi_l = p["phi_l"]
    omega_max = p["omega_max"]

    num1 = (0.5 * fcl * (1.0 - np.sin(phi_l)) - sy * np.sin(phi_l)) * np.cos(omega_max)
    num2 = (0.5 * fce * (1.0 - np.cos(phi_l)) - sx * np.cos(phi_l)) * np.sin(omega_max)
    den  = np.cos(omega_max - phi_l)

    if abs(den) < 1e-12:
        return np.inf

    return (num1 + num2) / den

def yield_components(sx, sy, txy, p):
    ftx = p["ftx"]
    fcm = p["fcm"]
    nu  = p["nu"]

    # I
    F1 = txy**2 + (ftx - sx) * sy

    # II
    F2 = txy**2 - (fcm + sx) * (fcm + sy)

    # VIII
    tau_viii = 0.5 * nu * fcm
    F3 = abs(txy) - tau_viii

    # X
    tau_x = tau_X(sx, p)
    F4 = abs(txy) - tau_x

    # XI
    tau_xi = tau_XI(sx, sy, p)
    F5 = abs(txy) - tau_xi

    return F1, F2, F3, F4, F5


def yield_function(sx, sy, txy, p):
    return max(yield_components(sx, sy, txy, p))

# symmetri
def swap_sigma1_sigma2(data):
    return data[:, [1, 0]]

page_plot_data = {
    0.0: np.vstack([
        page_data[0.0],
        swap_sigma1_sigma2(page_data[90.0])
    ]),
    22.5: np.vstack([
        page_data[22.5],
        swap_sigma1_sigma2(page_data[67.5])
    ]),
    45.0: np.vstack([
        page_data[45.0],
        swap_sigma1_sigma2(page_data[45.0])
    ]),
}

def yield_in_principal_plane(sig1, sig2, theta_deg, p):
    sx, sy, txy = principal_to_material(sig1, sig2, theta_deg)
    return yield_function(sx, sy, txy, p)


def yield_components_in_principal_plane(sig1, sig2, theta_deg, p):
    sx, sy, txy = principal_to_material(sig1, sig2, theta_deg)
    return yield_components(sx, sy, txy, p)

# plot en figur

def plot_individual_regions(theta_deg, p, exp_data=None, ax=None,
                            xlim=(-12, 2), ylim=(-12, 2), ngrid=300):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    s1_vals = np.linspace(xlim[0], xlim[1], ngrid)
    s2_vals = np.linspace(ylim[0], ylim[1], ngrid)
    S1, S2 = np.meshgrid(s1_vals, s2_vals)

    FI = np.empty_like(S1)
    FII = np.empty_like(S1)
    FVIII = np.empty_like(S1)
    FX = np.empty_like(S1)
    FXI = np.empty_like(S1)

    for i in range(S1.shape[0]):
        for j in range(S1.shape[1]):
            f1, f2, f3, f4, f5 = yield_components_in_principal_plane(
                S1[i, j], S2[i, j], theta_deg, p
            )
            FI[i, j] = f1
            FII[i, j] = f2
            FVIII[i, j] = f3
            FX[i, j] = f4
            FXI[i, j] = f5

    ax.contour(S1, S2, FI, levels=[0], linewidths=1.6, colors=['C0'])
    ax.contour(S1, S2, FII, levels=[0], linewidths=1.6, colors=['C1'])
    ax.contour(S1, S2, FVIII, levels=[0], linewidths=1.6, colors=['C2'])
    ax.contour(S1, S2, FX, levels=[0], linewidths=1.6, colors=['C3'])
    ax.contour(S1, S2, FXI, levels=[0], linewidths=1.6, colors=['C4'])

    ax.plot([], [], color='C0', lw=1.6, label='I')
    ax.plot([], [], color='C1', lw=1.6, label='II')
    ax.plot([], [], color='C2', lw=1.6, label='VIII')
    ax.plot([], [], color='C3', lw=1.6, label='X')
    ax.plot([], [], color='C4', lw=1.6, label='XI')

    if exp_data is not None and len(exp_data) > 0:
        ax.plot(exp_data[:, 0], exp_data[:, 1], 'k*', ms=5, label='Forsøg (Page)')

    ax.axhline(0, color='k', lw=0.8)
    ax.axvline(0, color='k', lw=0.8)
    ax.grid(True, linestyle='--', linewidth=0.8)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r'$\sigma_1$ [MPa]')
    ax.set_ylabel(r'$\sigma_2$ [MPa]')
    ax.set_title(rf'Enkeltflader for $\theta={theta_deg}^\circ$')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

    return ax


def plot_active_region_map(theta_deg, p, exp_data=None, ax=None,
                           xlim=(-12, 2), ylim=(-12, 2), ngrid=300):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    s1_vals = np.linspace(xlim[0], xlim[1], ngrid)
    s2_vals = np.linspace(ylim[0], ylim[1], ngrid)
    S1, S2 = np.meshgrid(s1_vals, s2_vals)

    Fmax = np.empty_like(S1)
    active = np.empty_like(S1, dtype=int)

    for i in range(S1.shape[0]):
        for j in range(S1.shape[1]):
            vals = np.array(
                yield_components_in_principal_plane(S1[i, j], S2[i, j], theta_deg, p)
            )
            active[i, j] = np.argmax(vals)
            Fmax[i, j] = np.max(vals)

    ax.pcolormesh(S1, S2, active, shading='auto', cmap='tab10', alpha=0.35)
    ax.contour(S1, S2, Fmax, levels=[0], colors='k', linewidths=2.0)

    ax.plot([], [], color='C0', lw=8, alpha=0.35, label='I aktiv')
    ax.plot([], [], color='C1', lw=8, alpha=0.35, label='II aktiv')
    ax.plot([], [], color='C2', lw=8, alpha=0.35, label='VIII aktiv')
    ax.plot([], [], color='C3', lw=8, alpha=0.35, label='X aktiv')
    ax.plot([], [], color='C4', lw=8, alpha=0.35, label='XI aktiv')
    ax.plot([], [], color='k', lw=2.0, label='Samlet flydeflade')

    if exp_data is not None and len(exp_data) > 0:
        ax.plot(exp_data[:, 0], exp_data[:, 1], 'k*', ms=5, label='Forsøg (Page)')

    ax.axhline(0, color='k', lw=0.8)
    ax.axvline(0, color='k', lw=0.8)
    ax.grid(True, linestyle='--', linewidth=0.8)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r'$\sigma_1$ [MPa]')
    ax.set_ylabel(r'$\sigma_2$ [MPa]')
    ax.set_title(rf'Aktiv region for $\theta={theta_deg}^\circ$')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

    return ax


def plot_region_diagnostics(theta=0.0):
    exp_data = page_plot_data.get(theta, None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    plot_individual_regions(
        theta, params,
        exp_data=exp_data,
        ax=axes[0],
        xlim=(-12, 2),
        ylim=(-12, 2),
        ngrid=250
    )

    plot_active_region_map(
        theta, params,
        exp_data=exp_data,
        ax=axes[1],
        xlim=(-12, 2),
        ylim=(-12, 2),
        ngrid=250
    )

    plt.tight_layout()
    plt.show()



def plot_comparison(theta_deg, p, exp_data=None, ax=None,
                    xlim=(-12, 2), ylim=(-12, 2), ngrid=600):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # grid in sigma1-sigma2 plane
    s1_vals = np.linspace(xlim[0], xlim[1], ngrid)
    s2_vals = np.linspace(ylim[0], ylim[1], ngrid)
    S1, S2 = np.meshgrid(s1_vals, s2_vals)

    # evaluate yield function on grid
    F = np.empty_like(S1)

    for i in range(S1.shape[0]):
        for j in range(S1.shape[1]):
            F[i, j] = yield_in_principal_plane(S1[i, j], S2[i, j], theta_deg, p)

    # zero contour = yield surface
    cs = ax.contour(S1, S2, F, levels=[0.0], colors=['C0'], linewidths=1.8)

    # proxy line for legend
    if len(cs.allsegs) > 0 and len(cs.allsegs[0]) > 0:
        ax.plot([], [], color='C0', lw=1.8, label='Simplificeret flydeflade')

    # experimental data
    if exp_data is not None and len(exp_data) > 0:
        ax.plot(exp_data[:, 0], exp_data[:, 1], 'k*', ms=5, label='Forsøgsresultater (Page)')

    ax.axhline(0, color='k', lw=0.8)
    ax.axvline(0, color='k', lw=0.8)
    ax.grid(True, linestyle='--', linewidth=0.8)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r'$\sigma_1$ [MPa]')
    ax.set_ylabel(r'$\sigma_2$ [MPa]')
    ax.set_title(rf'Sammenligning mellem teori og forsøg for $\theta={theta_deg}^\circ$')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

    return ax

def plot_all():
    thetas = [0.0, 22.5, 45.0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, theta in zip(axes, thetas):
        exp_data = page_plot_data.get(theta, None)
        plot_comparison(
            theta, params,
            exp_data=exp_data,
            ax=ax,
            xlim=(-12, 2),
            ylim=(-12, 2),
            ngrid=500
        )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_all()

if __name__ == "__main__":
    plot_region_diagnostics(theta=45.0)