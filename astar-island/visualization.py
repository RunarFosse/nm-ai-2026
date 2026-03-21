import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def visualize(grids, h, w):
    value_to_color = {
        0: 'black',
        1: 'orange',
        2: 'cyan',
        3: '#8B0000',
        4: '#006400',
        5: 'gray',
        10: 'blue',
        11: 'beige'
    }

    values = sorted(value_to_color.keys())
    colors = [value_to_color[v] for v in values]

    cmap = ListedColormap(colors)
    bounds = [v - 0.5 for v in values] + [values[-1] + 0.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(h, w, figsize=(10, 7))
    axes = np.atleast_1d(axes).flatten()

    for idx, ax in enumerate(axes):
        if idx < len(grids):
            grid = np.array(grids[idx])  # ← FIX HERE

            ax.imshow(grid, cmap=cmap, norm=norm)

            rows, cols = grid.shape
            ax.set_xticks(np.arange(-0.5, cols, 1))
            ax.set_yticks(np.arange(-0.5, rows, 1))
            ax.grid(which='major', color='black', linestyle='-', linewidth=0.2)

            ax.set_xticklabels([])
            ax.set_yticklabels([])

            ax.set_title(f"SEED {idx+1}")
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()