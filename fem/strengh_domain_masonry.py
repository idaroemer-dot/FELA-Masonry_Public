import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# material parameters
# -----------------------------

# fcm = 5.0
# ftx = 0.3
# nu  = 0.7

# fcl = 1.0
# fcs = 1.0
# fce = 1.0

# phi_l = np.deg2rad(30)
# phi_s = np.deg2rad(30)
# phi_e = np.deg2rad(30)

# tau_cap = 0.5 * nu * fcm

# # -----------------------------
# # stress grid
# # -----------------------------

# sx = np.linspace(-fcm, ftx, 250)
# sy = np.linspace(-fcm, 0, 250)

# SX, SY = np.meshgrid(sx, sy)

# # -----------------------------
# # Region I - tension failure parallel to bed joint
# # -----------------------------

# tau_I = np.sqrt(np.maximum(-(ftx - SX) * SY,0))

# # -----------------------------
# # Region II - compression failure masonry
# # -----------------------------

# tau_II = np.sqrt(np.maximum((fcm + SX)*(fcm + SY),0))

# # -----------------------------
# # Region IVa - compression-shear cylinder
# # -----------------------------

# tau_IV = np.sqrt(np.maximum(-(SY*(fcm + SY)),0))

# # -----------------------------
# # Region VIII - shear capacity limit
# # -----------------------------

# tau_VIII = tau_cap*np.ones_like(SX)

# # -----------------------------
# # Region IX - bed joint failure
# # -----------------------------

# tau_IX = (
#     0.5*fcl*(1-np.sin(phi_l))/np.cos(phi_l)
#     - SY*np.tan(phi_l)
# )

# tau_IX = np.maximum(tau_IX,0)

# # -----------------------------
# # Region X - head joint failure
# # -----------------------------

# tau_X = (
#     0.5*fcs*(1-np.sin(phi_s))/np.cos(phi_s)
#     - SX*np.tan(phi_s)
# )

# tau_X = np.maximum(tau_X,0)

# # -----------------------------
# # Region XI - staircase failure
# # -----------------------------

# omega = np.deg2rad(45)

# tau_XI = (
#     0.5*fcl*np.cos(omega)*(1-np.sin(phi_l))
#     +0.5*fce*np.sin(omega)*(1-np.cos(phi_l))
#     -SX*np.sin(omega)*np.cos(phi_l)
#     -SY*np.cos(omega)*np.sin(phi_l)
# )

# tau_XI = np.maximum(tau_XI,0)

# # -----------------------------
# # combine all surfaces
# # -----------------------------

# tau_max = np.minimum.reduce([
#     tau_I,
#     tau_II,
#     tau_IV,
#     tau_VIII,
#     tau_IX,
#     tau_X,
#     tau_XI
# ])

# # -----------------------------
# # plotting
# # -----------------------------

# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(projection='3d')

# ax.plot_surface(SX, SY, tau_max, alpha=0.9)
# ax.plot_surface(SX, SY, -tau_max, alpha=0.9)

# ax.set_xlabel(r'$\sigma_x$')
# ax.set_ylabel(r'$\sigma_y$')
# ax.set_zlabel(r'$\tau_{xy}$')

# ax.set_title("Full Findsen masonry yield surface")

# ax.view_init(elev=25, azim=-120)
# ax.set_box_aspect([1,1,0.7])

# plt.show()



# -----------------------------
# material parameters - simple version
# -----------------------------

fcm = 15
ftm = 1.0
km  = 6

# -----------------------------
# stress grid
# -----------------------------

sx = np.linspace(-fcm, ftm, 250)
sy = np.linspace(-fcm, 0, 250)

SX, SY = np.meshgrid(sx, sy)

# -----------------------------
# Region I - tension failure
# -----------------------------

tau_I = np.sqrt(np.maximum((ftm - SX)*(ftm - SY),0))

# -----------------------------
# Region II - compression failure 
# -----------------------------

tau_II = np.sqrt(np.maximum((fcm + SX)*(fcm + SY),0))

# -----------------------------
# Region III - sliding failure
# -----------------------------

tau_III = np.sqrt(
    np.maximum(
        (1/(1+km)**2) *
        (fcm - km*SX + SY) *
        (fcm - km*SY + SX),
        0
    )
)

# -----------------------------
# combine all surfaces
# -----------------------------

tau_max = np.minimum.reduce([
    tau_I,
    tau_II,
    tau_III
])

# -----------------------------
# plotting
# -----------------------------

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d')
colors = ["#4F5150", "#D7DDD3"]

ax.plot_surface(SX, SY, tau_max, alpha=0.9, color=colors[1])
ax.plot_surface(SX, SY, -tau_max, alpha=0.9, color=colors[0])

ax.set_xlabel(r'$\sigma_x$')
ax.set_ylabel(r'$\sigma_y$')
ax.set_zlabel(r'$\tau_{xy}$')

ax.set_title("Simple Findsen masonry yield surface")

ax.view_init(elev=25, azim=-120)
ax.set_box_aspect([1,1,0.7])

plt.show()

