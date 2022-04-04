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
            raise Exception('[COLOURMAP] Error: seaborn is missing! Try to: pip install seaborn')
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
    if isinstance(colors, list):
        colors = np.array(colors)
    if len(colors.shape)==1:
        colors = np.array([colors])

    if not keep_alpha:
        colors = colors[:, 0:3]
    hexcolors = list(map(lambda x: matplotlib.colors.to_hex(x, keep_alpha=keep_alpha), colors))
    return np.array(hexcolors)


# %%
def _rgb2hex(c_rgb):
    """RGB to Hex.

    Parameters
    ----------
    c_rgb : tuple (255, 255, 255)
        rgb colors in range 0-255.

    Returns
    -------
    str.
        Hex color.

    Examples
    --------
    hexcolor = _rgb2hex([255,255,255])
    """
    # Components need to be integers for hex to make sense
    c_rgb = [int(x) for x in c_rgb]
    return "#"+"".join(["0{0:x}".format(v) if v < 16 else
              "{0:x}".format(v) for v in c_rgb])


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
    if 'str' in str(type(colors)):
        colors = np.array([colors])

    rgbcolors = list(map(lambda x: matplotlib.colors.to_rgb(x), colors))
    return np.array(rgbcolors)


def _hex2rgb(c_hex):
    """Convert hex to rgb.

    Parameters
    ----------
    c_hex : str
        Hex color.

    Returns
    -------
    list
        RGB color.

    Examples
    --------
    rgbcolor = _hex2rgb("#FFFFFF")

    """
    # Pass 16 to the integer function for change of base
    return [int(c_hex[i:i + 2], 16) for i in range(1, 6, 2)]


# %%
def fromlist(y, cmap='Set1', gradient=None, method='matplotlib', scheme='rgb'):
    """Generate colors from input list.

    Description
    ------------
    This function creates unique colors based on the input list y and the cmap.
    When the gradient hex color is defined, such as '#000000', a gradient coloring space is created between two colors.
        The start color of the particular y, using the cmap and
        The end color is the defined gradient, such as '#000000'.

    Parameters
    ----------
    y : list of strings or integers
        For each unique value, a unique color is given back.
    cmap : String, optional
        Colormap. The default is 'Set1'.
    scheme : String, optional
        The output of color is in the scheme:
        'rgb'
        'hex'
    gradient : String, (default: None)
        Hex end color for the gradient.
        '#FFFFFF'
    method : String, optional
        Method to generate colors
        'matplotlib' (default)
        'seaborn'

    Returns
    -------
    tuple containing:
        List of colors in the same order as y.
        dict for the unique colors

    References
    ----------
        * https://github.com/bsouthga/blog/blob/master/public/posts/color-gradients-with-python.md

    """
    # make unique
    y = np.array(y)
    uiy=np.unique(y)
    # Get colors
    getcolors=generate(len(uiy), cmap=cmap, method=method)
    # Make dict for each search
    colordict=dict(zip(uiy, getcolors))

    # Color using gradient.
    if gradient is not None:
        rgb_colors = np.array([[0.0, 0.0, 0.0]] * len(y))
        hex_colors = np.array(['#000000'] * len(y))
        for i, _ in enumerate(uiy):
            Iloc = uiy[i]==y
            c_gradient = linear_gradient(_rgb2hex(colordict.get(uiy[i]) * 255), finish_hex=gradient, n=sum(Iloc))
            rgb_gradient = np.c_[c_gradient['r'], c_gradient['g'], c_gradient['b']]
            hex_gradient = np.array(c_gradient['hex'])
            rgb_colors[Iloc] = rgb_gradient / 255
            hex_colors[Iloc] = hex_gradient
    else:
        # Get colors for y
        rgb_colors=list(map(colordict.get, y))
        # Stack list of arrays into single array
        rgb_colors = np.vstack(rgb_colors)

    # Set the output coloring scheme
    if scheme=='hex':
        colors = rgb2hex(rgb_colors)
    else:
        colors = rgb_colors
    # Return
    return(colors, colordict)


# %%
def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    """Return a gradient list of (n) colors between two hex colors.

    Description
    -----------
    start_hex and finish_hex should be the full six-digit color string, inlcuding the number sign ("#FFFFFF")

    Parameters
    ----------
    start_hex : str
        Hex starting color.
    finish_hex : str, optional
        Hex end color. The default is "#FFFFFF".
    n : int, (default: 10)
        Spacing between start-stop colors.

    Returns
    -------
    dict
        lineair spacing.

    """
    # Starting and ending colors in RGB form
    s = _hex2rgb(start_hex)
    f = _hex2rgb(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
      # Interpolate RGB vector for color at the current value of t
      curr_vector = [
        int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
        for j in range(3)
      ]
      # Add it to our list of output colors
      RGB_list.append(curr_vector)

    return _color_dict(RGB_list)


def _color_dict(gradient):
    """Color to dictionary.

    Description
    -----------
    Takes in a list of RGB sub-lists and returns dictionary of colors in RGB and
    hex form for use in a graphing function defined later on.

    """
    return {"hex":[_rgb2hex(RGB) for RGB in gradient],
        "r":[RGB[0] for RGB in gradient],
        "g":[RGB[1] for RGB in gradient],
        "b":[RGB[2] for RGB in gradient]}
# %%