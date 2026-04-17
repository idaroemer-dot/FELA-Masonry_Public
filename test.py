import numpy as np
import matplotlib.pyplot as plt

# Create grid in x2, x3
x2 = np.linspace(-5, 5, 100)
x3 = np.linspace(-5, 5, 100)
X2, X3 = np.meshgrid(x2, x3)

# Compute x1
X1 = np.sqrt(X2**2 + X3**2)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X1, X2, X3)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')

plt.show()
