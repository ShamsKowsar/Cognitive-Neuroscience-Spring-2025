
from menpo.feature import normalize
from menpo.visualize import print_progress
from menpofit.aam import HolisticAAM
from menpofit.aam import LucasKanadeAAMFitter
from pathlib import Path
import matplotlib.pyplot as plt
import menpo.io as mio
import numpy as np
import pickle

# Put this aam.pkl file in the folder you've shared with this docker container
# Possibly: /workspace/shared_with_host or /home/shared_with_host
with open('/home/data/aam.pkl', 'rb') as f:
    aam = pickle.load(f)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))



    
hex_color = "#7f7f7f"  
rgb_color = hex_to_rgb(hex_color)

def get_sample(weights_shape,weights_appearance,save=False,save_path=None):
    res=aam.instance(weights_shape,weights_appearance)
    
    img=res.as_masked().pixels
    img = np.transpose(img, (1, 2, 0))
    if img.dtype == np.float32 or img.dtype == np.float64:
        img_min = img.min()
        img_max = img.max()
        if img_max > 1.0 or img_min < 0.0:  # Check if normalization is needed
            img = (img - img_min) / (img_max - img_min)  # Normalize to [0, 1]

    if save:
        plt.imsave(save_path,img)
    return img



def interpolate_points(p1, p2, n):
    """
    Generate `n` points between two 20-dimensional points `p1` and `p2`.

    Parameters:
    p1 (array-like): First 20-dimensional point.
    p2 (array-like): Second 20-dimensional point.
    n (int): Number of points to generate.

    Returns:
    numpy.ndarray: Array of shape (n, 20) containing the interpolated points.
    """
    return np.linspace(p1, p2, num=n, endpoint=False)




def random_points_on_hypersphere(center, radius, num_points=10):
    dim = len(center)  # Get the dimensionality (20 in this case)
    
    # Generate random points from a normal distribution
    random_directions = np.random.randn(num_points, dim)
    
    # Normalize them to lie on the unit hypersphere
    unit_vectors = random_directions / np.linalg.norm(random_directions, axis=1, keepdims=True)
    
    # Scale by the desired radius and shift by the center
    points = center + radius * unit_vectors
    
    return points


