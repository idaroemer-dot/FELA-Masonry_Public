import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# material parameters
# -----------------------------

fcm = 5.0
ftx = 0.1
nu  = 0.7

fcl = 1.0
fcs = 1.0
fce = 1.0

phi_l = np.deg2rad(30)
phi_s = np.deg2rad(30)
phi_e = np.deg2rad(30)

tau_cap = 0.5 * nu * fcm

# # -----------------------------
# # stress grid
# # -----------------------------

sx = np.linspace(-fcm, ftx, 250)
sy = np.linspace(-fcm, 0, 250)

SX, SY = np.meshgrid(sx, sy)

# # -----------------------------
# # Region I - tension failure 
# # -----------------------------

tau_I = np.sqrt(np.maximum(-(ftx - SX) * SY,0))

# colors = ["#FF00FF", "#FFB4F7"]
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(projection='3d')

# ax.plot_surface(SX, SY, tau_I, alpha=0.9, color=colors[0])
# ax.plot_surface(SX, SY, -tau_I, alpha=0.9, color=colors[1])

# ax.set_xlabel(r'$\sigma_x$')
# ax.set_ylabel(r'$\sigma_y$')
# ax.set_zlabel(r'$\tau_{xy}$')

# ax.set_title("Region I - tension failure")

# ax.view_init(elev=25, azim=-120)
# ax.set_box_aspect([1,1,1])

# plt.show()

# # -----------------------------
# # Region II - compression failure masonry
# # -----------------------------

tau_II = np.sqrt(np.maximum((fcm + SX)*(fcm + SY),0))

# colors = ["#E5FF00", "#FAFFB4"]
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(projection='3d')

# ax.plot_surface(SX, SY, tau_II, alpha=0.9, color=colors[0])
# ax.plot_surface(SX, SY, -tau_II, alpha=0.9, color=colors[1])

# ax.set_xlabel(r'$\sigma_x$')
# ax.set_ylabel(r'$\sigma_y$')
# ax.set_zlabel(r'$\tau_{xy}$')

# ax.set_title("Region II - compression failure")

# ax.view_init(elev=25, azim=-120)
# ax.set_box_aspect([1,1,1])

# plt.show()

# # -----------------------------
# # Region IVa - compression-shear cylinder
# # -----------------------------

tau_IVa = np.sqrt(np.maximum(-(SY*(fcm + SY)),0))

# colors = ["#FF9100", "#FFDFB4"]
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(projection='3d')

# ax.plot_surface(SX, SY, tau_IV, alpha=0.9, color=colors[0])
# ax.plot_surface(SX, SY, -tau_IV, alpha=0.9, color=colors[1])

# ax.set_xlabel(r'$\sigma_x$')
# ax.set_ylabel(r'$\sigma_y$')
# ax.set_zlabel(r'$\tau_{xy}$')

# ax.set_title("Region IVa - compression-shear")

# ax.view_init(elev=25, azim=-120)
# ax.set_box_aspect([1,1,1])

# plt.show()

# # -----------------------------
# # Region VIII - shear capacity limit
# # -----------------------------

tau_VIII = tau_cap*np.ones_like(SX)


# colors = ["#FF8359", "#FFC7B4"]
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(projection='3d')

# ax.plot_surface(SX, SY, tau_VIII, alpha=0.9, color=colors[0])
# ax.plot_surface(SX, SY, -tau_VIII, alpha=0.9, color=colors[1])

# ax.set_xlabel(r'$\sigma_x$')
# ax.set_ylabel(r'$\sigma_y$')
# ax.set_zlabel(r'$\tau_{xy}$')

# ax.set_title("Region VIII - shear capacity limit")

# ax.view_init(elev=25, azim=-120)
# ax.set_box_aspect([1,1,1])

# plt.show()


# # -----------------------------
# # Region IX - bed joint failure
# # -----------------------------

# tau_IX = (
#     0.5*fcl*(1-np.sin(phi_l))/np.cos(phi_l)
#     - SY*np.tan(phi_l)
# )
alpha = np.asin(2*SY/fcl+1)

tau_IX = (
    0.5*fcl*(1-np.sin(alpha))/np.cos(alpha)
    - SY*np.tan(alpha)
)

tau_IX = np.maximum(tau_IX,0)

# colors = ["#2BFF00", "#BFFFB4"]
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(projection='3d')

# ax.plot_surface(SX, SY, tau_IX, alpha=0.9, color=colors[0])
# ax.plot_surface(SX, SY, -tau_IX, alpha=0.9, color=colors[1])

# ax.set_xlabel(r'$\sigma_x$')
# ax.set_ylabel(r'$\sigma_y$')
# ax.set_zlabel(r'$\tau_{xy}$')

# ax.set_title("Region IX - bed joint failure")

# ax.view_init(elev=25, azim=-120)
# ax.set_box_aspect([1,1,1])

# plt.show()

# # -----------------------------
# # Region X - head joint failure
# # -----------------------------
xi = 0.2
ms = 1.0
ls = 1.0
alpha_s = np.asin((SX+1/2*fcs*xi*ms+1/2*fce*(1-xi))/1/2*fcs*xi*ls+1/2*fce*(1-xi))

tau_X = (
    0.5*fcs*(1-np.sin(alpha_s))/np.cos(alpha_s)
    - SX*np.tan(alpha_s)
)

tau_X = np.maximum(tau_X,0)

# colors = ["#00FBFF", "#B4FFFF"]
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(projection='3d')

# ax.plot_surface(SX, SY, tau_X, alpha=0.9, color=colors[0])
# ax.plot_surface(SX, SY, -tau_X, alpha=0.9, color=colors[1])

# ax.set_xlabel(r'$\sigma_x$')
# ax.set_ylabel(r'$\sigma_y$')
# ax.set_zlabel(r'$\tau_{xy}$')

# ax.set_title("Region X - head joint failure")

# ax.view_init(elev=25, azim=-120)
# ax.set_box_aspect([1,1,1])

# plt.show()



# # -----------------------------
# # Region XI - staircase failure
# # -----------------------------

omega = np.deg2rad(45)

tau_XI = (
    0.5*fcl*np.cos(omega)*(1-np.sin(phi_l))
    +0.5*fce*np.sin(omega)*(1-np.cos(phi_l))
    -SX*np.sin(omega)*np.cos(phi_l)
    -SY*np.cos(omega)*np.sin(phi_l)
)

tau_XI = np.maximum(tau_XI,0)

# colors = ["#C800FF", "#D8B4FF"]
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(projection='3d')

# ax.plot_surface(SX, SY, tau_XI, alpha=0.9, color=colors[0])
# ax.plot_surface(SX, SY, -tau_XI, alpha=0.9, color=colors[1])

# ax.set_xlabel(r'$\sigma_x$')
# ax.set_ylabel(r'$\sigma_y$')
# ax.set_zlabel(r'$\tau_{xy}$')

# ax.set_title("Region XI - staircase failure")

# ax.view_init(elev=25, azim=-120)
# ax.set_box_aspect([1,1,1])

# plt.show()

# # -----------------------------
# # combine all surfaces
# # -----------------------------

tau_max = np.minimum.reduce([
    #tau_I,
    #tau_II,
   # tau_IVa,
   # tau_VIII,
   #tau_IX,
    tau_X,
   # tau_XI
])

# -----------------------------
# plotting
# -----------------------------
colors = ["#708D74FF", "#CCFDD6"]
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d')

ax.plot_surface(SX, SY, tau_max, alpha=0.9, color=colors[0])
ax.plot_surface(SX, SY, -tau_max, alpha=0.9, color=colors[1])

ax.set_xlabel(r'$\sigma_x$')
ax.set_ylabel(r'$\sigma_y$')
ax.set_zlabel(r'$\tau_{xy}$')

ax.set_title(" masonry yield surface")

ax.view_init(elev=25, azim=-120)
ax.set_box_aspect([1,1,0.7])

plt.show()



# # -----------------------------
# # material parameters - simple version
# # -----------------------------

# fcm = 15
# ftm = 1.0
# km  = 6

# # -----------------------------
# # stress grid
# # -----------------------------

# sx = np.linspace(-fcm, ftm, 250)
# sy = np.linspace(-fcm, 0, 250)

# SX, SY = np.meshgrid(sx, sy)

# # -----------------------------
# # Region I - tension failure
# # -----------------------------

# tau_I = np.sqrt(np.maximum((ftm - SX)*(ftm - SY),0))

# # -----------------------------
# # Region II - compression failure 
# # -----------------------------

# tau_II = np.sqrt(np.maximum((fcm + SX)*(fcm + SY),0))

# # -----------------------------
# # Region III - sliding failure
# # -----------------------------

# tau_III = np.sqrt(
#     np.maximum(
#         (1/(1+km)**2) *
#         (fcm - km*SX + SY) *
#         (fcm - km*SY + SX),
#         0
#     )
# )

# # -----------------------------
# # combine all surfaces
# # -----------------------------

# tau_max = np.minimum.reduce([
#     tau_I,
#     tau_II,
#     tau_III
# ])

# -----------------------------
# plotting
# -----------------------------

# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(projection='3d')
# colors = ["#4F5150", "#D7DDD3"]

# ax.plot_surface(SX, SY, tau_max, alpha=0.9, color=colors[1])
# ax.plot_surface(SX, SY, -tau_max, alpha=0.9, color=colors[0])

# ax.set_xlabel(r'$\sigma_x$')
# ax.set_ylabel(r'$\sigma_y$')
# ax.set_zlabel(r'$\tau_{xy}$')

# ax.set_title("Simple Findsen masonry yield surface")

# ax.view_init(elev=25, azim=-120)
# ax.set_box_aspect([1,1,0.7])

# plt.show()
