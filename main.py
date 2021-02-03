import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from PIL import Image
import sys
import math
import os

#*Parameters___________________________________________________________________________________
#The display colour for each saved image
COLOUR = "plasma"
#Size (diameter) in pixels of each filter pass
FILTER_SIZE = 40
#for 2 (or more) passes, the modeled filter is larger than the seperate filters
TARGET_SIZE = 2*(FILTER_SIZE-1) + 1
#Pixels with a lower magnitude than THRESHOLD aren't included in the kernel
THRESHOLD = 0.04
#Gradient descent learning rate
LEARNING_RATE = 0.5
#amount of momentum based gradient descent
BETA = 0.85
#Number of descent iterations
N_ITERS = 3000

#Not really a parameter, it would take some work to get this working for more than 2 filters
FILTERS = 2

#*Target filter________________________________________________________________________________
#Stolen circle code, where ratio corrects for smaller individual filter sizes compared to the modeled one
x, y = np.meshgrid(np.linspace(-1, 1, TARGET_SIZE), np.linspace(-1, 1, TARGET_SIZE))
ratio = FILTER_SIZE/TARGET_SIZE
circle = np.array(np.clip((0.825*ratio - np.sqrt(x*x + y*y)) * 12.0/ratio, 0.0, 1.0), dtype='float64')

REF_SHAPE = circle

#*Optimisation_________________________________________________________________________________
#Convolve the filters together to find out what shape they are seperating
def model(vs):
    return jax.scipy.signal.convolve2d(vs[0],vs[1])

#Loss for how close to the target the modeled filter is
def filter_loss(mdld):
    return jnp.mean(jnp.square(mdld-REF_SHAPE))

#Cheap symmetry loss
def sym_loss(mdld):
    loss  = jnp.mean(jnp.square( mdld - jnp.rot90(mdld,1) ))
    loss += jnp.mean(jnp.square( mdld - jnp.rot90(mdld,2) ))
    loss += jnp.mean(jnp.square( mdld - jnp.rot90(mdld,3) ))
    return loss/3

#Loss for how many pixels are being sampled in both filters
#I found that continuity is very important so sigmoid is good here
def perf_loss(vs):
    total = 0
    for i in range(FILTERS):
        x       = jnp.abs(vs[i]/THRESHOLD)-1
        sigmoid = 1 / (1 + jnp.exp(-x*1))
        total  += jnp.mean(sigmoid)
    return total/FILTERS

#Combine all loss functions with some weights
def full_loss(vs, term1, term2, term3):
    mdld = model(vs)
    l1 = 3.0 * filter_loss(mdld)
    l2 = 1.0 * sym_loss(mdld)
    l3 = 1.0 * perf_loss(vs)
    return l1*term1+l2*term2+l3*term3

#Get the jax functions for better perf
grad = jax.jit(jax.grad(full_loss))
loss = jax.jit(full_loss)

#Optimises input filters to acheive REF_SHAPE
def optimize_loop(vs, ITERS):
    #Init velocity as identical to vs but with zeros
    velocity = []
    for i in range(FILTERS):
        velocity.append(np.zeros_like(vs[0]))

    for n in range(ITERS):
        #Print loss at 1% intervals
        if n % (ITERS/100) == 0:
            print(str(round(n/ITERS*100.0)) + "% :", round(loss(vs,1,0,0),5), round(loss(vs,0,1,0),5), round(loss(vs,0,0,1),5), round(loss(vs,1,1,1),5))

        #get gradients
        r1,r2,r3 = np.random.rand(), np.random.rand(), np.random.rand()
        length = np.sqrt(r1*r1+r2*r2+r3*r3)
        grads = grad(vs,r1/length,r2/length,r3/length)

        for i in range(FILTERS):
            #for vannila descent
            # vs[i] -= LEARNING_RATE*grads[i]

            #for spicy momentum descent
            velocity[i] = BETA*velocity[i] + LEARNING_RATE*grads[i]
            vs[i] -= velocity[i]
            
    return vs

#*Apply optimisation___________________________________________________________________________
# To start learning from scratch
vs_raw = []
for j in range(FILTERS):
    vs_raw.append(jnp.array(np.random.rand(FILTER_SIZE,FILTER_SIZE)*2.-1., dtype="float64") / jnp.sqrt(FILTER_SIZE*FILTER_SIZE))
vs_raw = optimize_loop(vs_raw, N_ITERS)

# #To load a previous filter
# vs_raw = []
# vs_raw.append(np.loadtxt("filter_0.txt", delimiter=",")+jnp.array(np.random.rand(FILTER_SIZE,FILTER_SIZE)*2.-1., dtype="float64")*0.01)
# vs_raw.append(np.loadtxt("filter_1.txt", delimiter=",")+jnp.array(np.random.rand(FILTER_SIZE,FILTER_SIZE)*2.-1., dtype="float64")*0.01)
# vs_raw = optimize_loop(vs_raw, N_ITERS)

#*Threshhold small magnitudes for modeled filter ______________________________________________
pix_sum      = 0
vs_threshold = []
for i in range(FILTERS):
    pix = (np.abs(vs_raw[i])>THRESHOLD)
    vs_threshold.append(pix * vs_raw[i])
    pix_sum += np.sum(pix)

print("-----------------------------------------------")
print("Pixel samples: " + str(pix_sum) )
print("raw loss:", round(loss(vs_raw,1,0,0),5), round(loss(vs_raw,0,1,0),5), round(loss(vs_raw,0,0,1),5), round(loss(vs_raw,1,1,1),5))
print("threshold loss:", round(loss(vs_threshold,1,0,0),5), round(loss(vs_threshold,0,1,0),5), round(loss(vs_threshold,0,0,1),5), round(loss(vs_threshold,1,1,1),5))

#*Save the kernel______________________________________________________________________________
#Create a folder for all kernels
save_folder = "kernels size " + str(FILTER_SIZE)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
#Create as subfolder for this kernel named [loss]
new_path = save_folder +"/" + str(round(loss(vs_threshold,1,1,1),5))
os.makedirs(new_path)
new_path += "/"

final_model = model(vs_threshold)

#save some visual results for inspection
plt.imsave(new_path+"target and solution.png", np.hstack((REF_SHAPE,final_model)), cmap=COLOUR)
plt.imsave(new_path+"both filters" + ".png", np.hstack(vs_threshold), cmap=COLOUR)

#normalize the filter
total = jnp.abs(np.sum(final_model))
vs_threshold /= np.power(total,1.0/FILTERS) #+0.0000001

#save the raw (not thresholded or normalized) filter as text (maybe your kernel was lucky and you want to optimise it further)
for i in range(FILTERS):
    np.savetxt(new_path+"filter_" + str(i) +".txt", np.asarray(vs_raw[i], dtype='float32'), delimiter=",")

#Process kernel into list of samples in the format (x,y,weight)
for i in range(FILTERS):
    kernel = ""
    for x in range(FILTER_SIZE):
        for y in range(FILTER_SIZE):
            val = vs_threshold[i][x][y]
            if val!=0:
                kernel += "vec3("+str(x-FILTER_SIZE/2) +","+str(y-FILTER_SIZE/2)+","+str(round(val,7))+"), " #export for glsl

    with open(new_path+"kernel_filter_"+str(i)+".txt", "w+") as file:
        file.write(kernel)
