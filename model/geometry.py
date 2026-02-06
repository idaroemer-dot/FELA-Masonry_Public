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


number_elements = 10

variables_per_element = np.zeros((number_elements, 9), dtype=int)
print(variables_per_element.shape)

for elements in range(number_elements):
    variables_per_element[elements, :] = elements * 9 + np.arange(9)   
number_variabels = 9 * number_elements

print(number_variabels)
print(variables_per_element)