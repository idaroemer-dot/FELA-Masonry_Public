import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Parameters
# -----------------------
mu = 0.5
cn = 0.1e6
ctau = 0.1e6
fy = 1.5e6

# -----------------------
# T2 Joints horizontal sliding
# -----------------------
# -----------------------
# Stress grid (3D plot)
# -----------------------
sx = np.linspace(-8e6, 2e6, 50)
sy = np.linspace(-8e6, 2e6, 50)
SX, SY = np.meshgrid(sx, sy)

# horizontal sliding planes
TAU_hh_pos = ctau - mu*SY
TAU_hh_neg = -ctau + mu*SY
colorhh = 'lightblue'

# σy = cn vertical plane
tau_dummy = np.linspace(-4e6, 4e6, 50)
SX2, TAU2 = np.meshgrid(sx, tau_dummy)
SY2 = np.ones_like(SX2) * cn


# # -----------------------
# # Create figure
# # -----------------------
# fig = plt.figure(figsize=(10, 14))

# # ==========================
# # FIRST PLOT (3D)
# # ==========================
# ax1 = fig.add_subplot(2, 1, 1, projection='3d')

# ax1.plot_surface(SX, SY, TAU_hh_pos, alpha=0.8, color=colorhh)
# ax1.plot_surface(SX, SY, TAU_hh_neg, alpha=0.8, color=colorhh)


# ax1.plot_surface(SX2, SY2, TAU2, alpha=0.8, color='lightcoral')

# ax1.set_xlabel(r'$\sigma_x$ [Pa]')
# ax1.set_ylabel(r'$\sigma_y$ [Pa]')
# ax1.set_zlabel(r'$\tau_{xy}$ [Pa]')
# ax1.set_title("3D Failure Planes")

# # ==========================
# # SECOND PLOT (2D slice σx = 0)
# # ==========================
# ax2 = fig.add_subplot(2, 1, 2)

# sigma_y = np.linspace(-8e6, 2e6, 200)
# tau_pos = ctau - mu*sigma_y
# tau_neg = -ctau + mu*sigma_y

# # Plot friction wedge
# ax2.plot(sigma_y, tau_pos, color = colorhh)
# ax2.plot(sigma_y, tau_neg, color = colorhh)

# # Plot σy = cn cutoff
# ax2.axvline(cn, color='lightcoral')

# # Fill admissible region
# ax2.fill_between(
#     sigma_y,
#     tau_neg,
#     tau_pos,
#     where=(sigma_y <= fy),
#     alpha=0.2,
# )

# ax2.set_xlabel(r'$\sigma_y$ [Pa]')
# ax2.set_ylabel(r'$\tau_{xy}$ [Pa]')
# ax2.set_title(r'2D Slice at $\sigma_x = 0$')

# ax2.set_xlim(-4e6, 4e6)
# ax2.set_ylim(-2.5e6, 2.5e6)
# ax2.grid(True)

# plt.subplots_adjust(hspace=0.4)
# plt.show()

# -----------------------
# T2 Joints diagonal
# -----------------------
hb = 0.05
lb = 0.25

#horizontal sliding
kh = 2*hb/lb
TAU_dh_pos = (
    ctau
    - mu*SY
    - kh*SX
    + kh*cn
) / (1 + mu*kh)

TAU_dh_neg = (
    -ctau
    + mu*SY
    + kh*SX
    - kh*cn
) / (1 + mu*kh)

#vertical sliding
colordv = 'lightgreen'
kv = lb/(2*hb)

TAU_dv_pos = (
    ctau
    - mu*SX
    - kv*SY
    + kv*cn
) / (1 + mu*kv)

TAU_dv_neg = (
    -ctau
    + mu*SX
    + kv*SY
    - kv*cn
) / (1 + mu*kv)

# -----------------------
# Create figure
# -----------------------
fig = plt.figure(figsize=(10, 14))

# ==========================
# FIRST PLOT (3D)
# ==========================
ax1 = fig.add_subplot(2, 1, 1, projection='3d')

ax1.plot_surface(SX, SY, TAU_dh_pos, alpha=0.8, color=colorhh)
ax1.plot_surface(SX, SY, TAU_dh_neg, alpha=0.8, color=colorhh)
ax1.plot_surface(SX, SY, TAU_dv_pos, alpha=0.8, color=colordv)
ax1.plot_surface(SX, SY, TAU_dv_neg, alpha=0.8, color=colordv)

ax1.set_xlabel(r'$\sigma_x$ [Pa]')
ax1.set_ylabel(r'$\sigma_y$ [Pa]')
ax1.set_zlabel(r'$\tau_{xy}$ [Pa]')
ax1.set_title("3D Failure Planes")

# ==========================
# SECOND PLOT (2D slice σx = 0)
# ==========================
ax2 = fig.add_subplot(2, 1, 2)

sigma_y = np.linspace(-8e6, 2e6, 400)

# --- Horizontal diagonal (σx = 0) ---
tau_dh_pos_2d = (
    ctau
    - mu*sigma_y
    + kh*cn
) / (1 + mu*kh)

tau_dh_neg_2d = (
    -ctau
    + mu*sigma_y
    - kh*cn
) / (1 + mu*kh)

# --- Vertical diagonal (σx = 0) ---
tau_dv_pos_2d = (
    ctau
    - kv*sigma_y
    + kv*cn
) / (1 + mu*kv)

tau_dv_neg_2d = (
    -ctau
    + kv*sigma_y
    - kv*cn
) / (1 + mu*kv)

# Plot horizontal sliding
ax2.plot(sigma_y, tau_dh_pos_2d, color=colorhh)
ax2.plot(sigma_y, tau_dh_neg_2d, color=colorhh)

# Plot vertical sliding
ax2.plot(sigma_y, tau_dv_pos_2d, color=colordv)
ax2.plot(sigma_y, tau_dv_neg_2d, color=colordv)

# Governing envelope (intersection of planes)
tau_upper = np.minimum(tau_dh_pos_2d, tau_dv_pos_2d)
tau_lower = np.maximum(tau_dh_neg_2d, tau_dv_neg_2d)

# Plot envelope (red curve like your image)
ax2.plot(sigma_y, tau_upper, color='red', linewidth=1, linestyle='--')
ax2.plot(sigma_y, tau_lower, color='red', linewidth=1, linestyle='--')

# Fill admissible region
ax2.fill_between(
    sigma_y,
    tau_lower,
    tau_upper,
    where=(tau_upper >= tau_lower),
    color='red',
    alpha=0.2,
)

ax2.set_xlabel(r'$\sigma_y$ [Pa]')
ax2.set_ylabel(r'$\tau_{xy}$ [Pa]')
ax2.set_title(r'2D Slice at $\sigma_x = 0$')

ax2.set_xlim(-4e6, 4e6)
ax2.set_ylim(-2.5e6, 2.5e6)
ax2.grid(True)

plt.subplots_adjust(hspace=0.4)
plt.show()

# -----------------------
# T2 Joints vertical
# -----------------------

# f = 0.5
# k = f * lb / (2 * hb)
# SX_v = cn - (SY * mu - ctau) * k

# tau = np.linspace(-10e6, 6e6, 60)
# SY3, TAU3 = np.meshgrid(sy, tau)
# SX3 = cn - (SY3 * mu - ctau) * k

# # -----------------------
# # Plot
# # -----------------------
# fig = plt.figure(figsize=(10, 12))

# # ========== 3D ==========
# ax1 = fig.add_subplot(2, 1, 1, projection="3d")
# ax1.plot_surface(SX3, SY3, TAU3, alpha=0.7)

# ax1.set_xlabel(r'$\sigma_x$ [Pa]')
# ax1.set_ylabel(r'$\sigma_y$ [Pa]')
# ax1.set_zlabel(r'$\tau_{xy}$ [Pa]')
# ax1.set_title(r'$\Phi_v = 0$ plane')

# # ========== 2D slice at tau_xy = 0 ==========
# ax2 = fig.add_subplot(2, 1, 2)

# sigma_y = np.linspace(-3e6, 2e6, 400)
# sigma_x_line = cn - (sigma_y * mu - ctau) * k   # σx(σy)
# ax2.plot(sigma_x_line, sigma_y, linewidth=3)

# # Shade admissible side:
# x_min, x_max = -3e6, 5e6
# ax2.fill_betweenx(
#     sigma_y,
#     x_min,                 
#     sigma_x_line,          
#     alpha=0.15
# )

# ax2.set_title(r'$\tau_{xy}=0$ MPa')
# ax2.set_xlabel(r'$\sigma_x$ [Pa]')
# ax2.set_ylabel(r'$\sigma_y$ [Pa]')
# ax2.set_xlim(x_min, x_max)
# ax2.set_ylim(-3e6, 2e6)
# ax2.grid(True)

# plt.subplots_adjust(hspace=0.35)
# plt.show()