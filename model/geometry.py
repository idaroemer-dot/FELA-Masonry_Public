import numpy as np 

L=1.0
H=1.0


ndivitionsx = 3
ndivitionsy = 3


#node coordinates: ncoor(node,:)=[x y] 
for x in np.linspace(0,L,ndivitionsx):
    for y in np.linspace(0,H,ndivitionsy):
        if 'ncoor' in locals():
            ncoor = np.vstack((ncoor,[x,y]))
        else:
            ncoor = np.array([[x,y]])


