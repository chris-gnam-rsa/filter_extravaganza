import numpy as np

def create_stars(num_stars):
    star_directions = np.random.randn(num_stars, 3)
    star_directions /= np.linalg.norm(star_directions, axis=1)[:, np.newaxis]
    return star_directions