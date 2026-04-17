import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Create grid
x = np.linspace(-1, 1, 200)
y = np.linspace(-1, 1, 200)
X, Y = np.meshgrid(x, y)

# Saddle surface
Z = X**2 - Y**2

# Custom colormap
colors = ["#4F5150", "#D7DDD3"]   
custom_cmap = LinearSegmentedColormap.from_list("mycmap", colors)

# Plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=custom_cmap, edgecolor='none')
ax.xaxis.line.set_color(colors[0])
ax.yaxis.line.set_color(colors[0])
ax.zaxis.line.set_color(colors[0])
ax.tick_params(axis='x', colors=colors[0])
ax.tick_params(axis='y', colors=colors[0])
ax.tick_params(axis='z', colors=colors[0])

plt.show()