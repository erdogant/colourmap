"""Python package colourmap generates an N unique colors from the specified input colormap."""

# Name        : colourmap.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Jan. 2020
# Licence     : MIT

# %% Libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# %% Main
def generate(N, cmap='Set1', method='matplotlib', keep_alpha=False):
    """Generate N RGB colors for cmap.

    Parameters
    ----------
    N : Integer
        Number of colors to be generated.
    cmap : String, optional
        'Set1'       (default)
        'Set2'
        'rainbow'
        'bwr'        Blue-white-red
        'binary' or 'binary_r'
        'seismic'    Blue-white-red
        'Blues'      white-to-blue
        'Reds'       white-to-red
        'Pastel1'    Discrete colors
        'Paired'     Discrete colors
        'Set1'       Discrete colors
    method : String, optional
        Method to generate colors
        'matplotlib' (default)
        'seaborn'

    References
    ----------
    Colormap: https://matplotlib.org/examples/color/colormaps_reference.html

    Returns
    -------
    color_list : numpy array with colors that range between [0-1, 0-1, 0-1].

    """
    if keep_alpha:
        listlen=4
    else:
        listlen=3

    if method=='seaborn':
        try:
            import seaborn as sns
        except:
            print('[COLOURMAP] Error: seaborn is missing! Try to: pip install seaborn')
        # color_list=sns.color_palette(cmap,N).as_hex()
        color_list=sns.color_palette(cmap, N)
    else:
        base = plt.cm.get_cmap(cmap)
        color_list = base(np.linspace(0, 1, N))[:, 0:listlen].tolist()
    return np.array(color_list)


# %%
def rgb2hex(colors, keep_alpha=False):
    """Convert RGB color-range to hex.

    Parameters
    ----------
    colors : list
        list of floats that range between [0-1, 0-1, 0-1].
    keep_alpha : bool, optional
        Keep the alpha value, which is the first number in RGB range. The default is False.

    Returns
    -------
    list of hex colors.

    """
    if not keep_alpha:
        colors = colors[:, 0:3]
    hexcolors = list(map(lambda x: matplotlib.colors.to_hex(x, keep_alpha=keep_alpha), colors))
    return np.array(hexcolors)


# %%
def hex2rgb(colors):
    """Convert hex color-range to RGB.

    Parameters
    ----------
    colors : list
        list of str.
    keep_alpha : bool, optional
        Keep the alpha value, which is the first number in RGB range. The default is False.

    Returns
    -------
    list of rgb colors.

    """
    rgbcolors = list(map(lambda x: matplotlib.colors.to_rgb(x), colors))
    return np.array(rgbcolors)


# %%
def fromlist(y, cmap='Set1', method='matplotlib'):
    """Generate colors from input list.

    Parameters
    ----------
    y : list of strings or integers
        For each unique value, a unique color is given back.
    cmap : String, optional
        Colormap. The default is 'Set1'.
    method : String, optional
        Method to generate colors
        'matplotlib' (default)
        'seaborn'


    Returns
    -------
    tuple containing:
        List of colors in the same order as y.
        dict for the unique colors

    """
    # make unique
    uiy=np.unique(y)
    # Get colors
    getcolors=generate(len(uiy), cmap=cmap, method=method)
    # Make dict for each search
    colordict=dict(zip(uiy, getcolors))
    # Get colors for y
    out=list(map(colordict.get, y))
    # Stack list of arrays into single array
    out = np.vstack(out)
    # Return
    return(out, colordict)
