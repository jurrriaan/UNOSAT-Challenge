import matplotlib
import matplotlib.pyplot as plt

def visualize_sample(np_arr, poly_mask, figsize=(10,10), linewidth=1):
    fig, ax = plt.subplots(figsize=figsize)
    ax.contour(poly_mask, 1, colors='red', linewidth=linewidth)
    ax.imshow(np_arr)
    plt.savefig('sample.png')