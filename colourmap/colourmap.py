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
def generate(N, cmap='Set1', method='matplotlib', keep_alpha=False, scheme='rgb', verbose=3):
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
    scheme : String, optional
        The output of color is in the scheme:
        'rgb'
        'hex'
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

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
        sns = _check_seaborn()
        color_list = sns.color_palette(cmap, N)
    else:
        # base = plt.cm.get_cmap(cmap)
        base = matplotlib.colormaps[cmap]
        color_list = base(np.linspace(0, 1, N))[:, 0:listlen].tolist()
        # If there are not enough colors in the cmap, use the seaborn method.
        uicolors = len(np.unique(rgb2hex(color_list)))
        if uicolors != N:
            if verbose>=2: print('[colourmap]> Warning: Colormap [%s] can not create [%d] unique colors! Available unique colors: [%d].' %(cmap, N, uicolors))

    # Set the output coloring scheme
    if scheme=='hex':
        colors = rgb2hex(color_list)
    else:
        colors = color_list

    return np.array(colors)


# %%
def _check_seaborn():
    try:
        import seaborn as sns
        return sns
    except:
        raise Exception('[colourmap]> Error: seaborn is missing! Try to: pip install seaborn')


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
def hex2rgba(colors):
    """Convert hex color-range to RGBA.

    Parameters
    ----------
    colors : list
        list of str.

    Returns
    -------
    list of rgb colors.

    """
    if 'str' in str(type(colors)):
        colors = np.array([colors])

    rgbcolors = list(map(lambda x: matplotlib.colors.to_rgba(x), colors))
    return np.array(rgbcolors)


# %%
def hex2rgb(colors):
    """Convert hex color-range to RGB.

    Parameters
    ----------
    colors : list
        list of str.

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
def fromlist(y, cmap='Set1', gradient=None, method='matplotlib', scheme='rgb', verbose=3):
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
    gradient : String, (default: None)
        Hex end color for the gradient.
        '#FFFFFF'
    method : String, optional
        Method to generate colors
        'matplotlib' (default)
        'seaborn'
    scheme : String, optional
        The output of color is in the scheme:
        'rgb'
        'hex'
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

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
    uiy = np.unique(y)

    # Get unique categories without sort
    # indexes = np.unique(y, return_index=True)[1]
    # uiy = [y[index] for index in sorted(indexes)]

    # Get colors
    colors_unique = generate(len(uiy), cmap=cmap, method=method, scheme=scheme, verbose=verbose)

    # Make dict for each search
    colordict = dict(zip(uiy, colors_unique))

    # Color using density and the gradient.
    if gradient is not None:
        # Set the scheme
        if scheme=='rgb':
            colors = np.array([[0.0, 0.0, 0.0]] * len(y))
        else:
            colors = np.array(['#000000'] * len(y))

        # Make the colors based on the density
        for i, _ in enumerate(uiy):
            Iloc = uiy[i]==y

            if scheme=='rgb':
                # Set the rgb colors
                c_gradient = linear_gradient(_rgb2hex(colordict.get(uiy[i]) * 255), finish_hex=gradient, n=sum(Iloc))
                colors[Iloc] = c_gradient['rgb'] / 255
            else:
                # Set the hex colors
                c_gradient = linear_gradient(colordict.get(uiy[i]), finish_hex=gradient, n=sum(Iloc))
                colors[Iloc] = np.array(c_gradient['hex'])
    else:
        # Get colors for y
        colors = list(map(colordict.get, y))

        # Stack list of arrays into single array
        if scheme=='rgb':
            colors = np.vstack(colors)
        else:
            colors = np.array(colors)

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
            int(s[j] + (float(t) / (n - 1)) * (f[j] - s[j]))
            for j in range(3)
            ]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)

    # convert to dict
    coldict = _color_dict(RGB_list)
    # return
    return coldict


def _color_dict(gradient):
    """Color to dictionary.

    Description
    -----------
    Takes in a list of RGB sub-lists and returns dictionary of colors in RGB and
    hex form for use in a graphing function defined later on.

    """
    hex_colors = [_rgb2hex(RGB) for RGB in gradient]
    rgb_colors = np.c_[[RGB[0] for RGB in gradient], [RGB[1] for RGB in gradient], [RGB[2] for RGB in gradient]]
    return {'hex': hex_colors, 'rgb': rgb_colors}
