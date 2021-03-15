import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def ant_model(bp, skeleton=None, skeleton_color=None, fr=0):
    fig, ax = plt.subplots(figsize=(3,4))
    # plot bodypoints
    ax.scatter(bp[fr,:,0], bp[fr,:,1], s=7, c='k')
    ax.set(title=f"Ant Model Bodypoints for Frame {fr}",
        xlabel="scaled x coord", ylabel="scaled y coord")
    # plot skeleton
    if (skeleton is not None) and (skeleton_color is not None):
        for skeleton_i, color_i in zip(skeleton, skeleton_color):
            ax.plot(bp[fr,skeleton_i,0], bp[fr,skeleton_i,1], alpha=0.7, c=color_i, linewidth=2)
    plt.show()
    return
