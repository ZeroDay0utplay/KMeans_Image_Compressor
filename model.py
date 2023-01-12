#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# # Read the image

# In[2]:


image = Image.open("src/me.png")


# In[3]:


img_mat = np.asanyarray(image) # convert it to 3D matrix (x,y,rgb) <-> rgb = (r,g,b)
img_mat = np.array([[pix[:3] for pix in r ] for r in img_mat])


# In[4]:


n, m, rgb = img_mat.shape


# ## Color Occurrences

# In[5]:


colors_occ = {}
for r in img_mat:
    for col in r:
        pix = " ".join([str(rgb) for rgb in col])
        if pix not in colors_occ:
            colors_occ[pix] = 0
        else:
            colors_occ[pix]+=1


# # Hex Values

# In[6]:


def hex_score(pixel):
    return sum(pixel)


# In[7]:


colors_values = {}
for r in img_mat:
    for col in r:
        pix = " ".join([str(rgb) for rgb in col])
        colors_values[pix] = hex_score(list(map(int, pix.split())))


# # My function Psy

# In[8]:


occ_mx = max(colors_occ.values())
occ_mn = min(colors_occ.values())
val_mx = max(colors_values.values())
val_mn = min(colors_values.values())


# In[9]:


def psy(color_occ, hex_value):
    # standarization 
    
    std_color_occ = (color_occ-occ_mn)/(occ_mx-occ_mn)
    std_color_value = (hex_value - val_mn)/(val_mx-val_mn)
    
    # No STD
    
    #std_color_occ = color_occ
    #std_color_value = hex_value
    
    if std_color_value == 0: return std_color_occ
    return std_color_occ - std_color_value


# In[10]:


my_fun = {k: psy(colors_occ[k], colors_values[k]) for k in colors_occ}


# ## Use highest psy score

# In[11]:


srt_psy = sorted(my_fun.items(), key=lambda x:x[1])
srt_psy = [list(map(int, elm[0].split())) for elm in srt_psy]
colors_16 = srt_psy[::-1][:16]
colors_16 = [pix[:3] for pix in colors_16]


# In[12]:


oc = sorted(colors_occ.items(), key=lambda x:x[1])
oc = [list(map(int, elm[0].split())) for elm in oc]
colors_16 = oc[::-1][:64]
colors_16 = [pix[:3] for pix in colors_16]


# In[13]:


def calc_dist(pixel, color):
    ln = 3
    dist = 0
    for i in range(ln):
        dist += (pixel[i]-color[i])**2
    return np.sqrt(dist)


# In[14]:


def rnd(x):
    if (int(x)-x) >=0.5:
        return int(x)+1
    return int(x)


# In[15]:


def calc_mean(vals, mat):
    mean = sum([mat[i[0],i[1]]/len(vals) for i in vals])
    return [rnd(x) for x in mean]


# In[17]:


def k_means(image, clusters):
    n, m, rgb_ = image.shape
    nb_clustrers = len(clusters)
    zdo = True
    while zdo:
        common_clusters = [[] for i in range(nb_clustrers)]
        zdo = False
        for i in range(n):
            for j in range(m):
                closer_dist = 1e9
                pix = list(image[i][j])
                ind=0
                for k in range(nb_clustrers):
                    if calc_dist(pix, clusters[k]) < closer_dist:
                        closer_dist = calc_dist(pix, clusters[k])
                        ind=k
                common_clusters[ind].append((i,j))

        for k in range(nb_clustrers):
            cmn = common_clusters[k]
            if len(cmn) == 0:
                continue
            if clusters[k] != calc_mean(cmn, image):
                zdo = True
            clusters[k] = calc_mean(cmn, image)
            for x in cmn:
                image[x[0], x[1]] = clusters[k]          
    return image


# In[18]:


image = img_mat.copy()
clusters = colors_16


# In[ ]:


img = k_means(image, colors_16)


# In[ ]:


new_image = Image.fromarray(img)


# In[ ]:


new_image.save("src/me2.png")


# In[ ]:




