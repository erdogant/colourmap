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
        try:
            base = matplotlib.colormaps[cmap]
        except:
            base = plt.cm.get_cmap(cmap)
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
def fromlist(y, X=None, cmap='Set1', gradient=None, method='matplotlib', scheme='rgb', verbose=3):
    """Generate colors from input list.

    This function creates unique colors based on the input list y and the cmap.
    When the gradient hex color is defined, such as '#000000', a gradient coloring space is created between two colors.
        The start color of the particular y, using the cmap and
        The end color is the defined gradient, such as '#000000'.

    Parameters
    ----------
    y : list of strings or integers
        For each unique value, a unique color is given back.
    X : numpy array (optional)
        Coordinates for the x-axis and y-axis: [x, y]. Should be the same length of y.
        If X is provided, the gradient will be based on the density.
    cmap : String, optional
        Colormap. The default is 'Set1'.
    gradient : String, (default: None)
        Hex color for the gradient based on the density.
        * None: Do not use gradient.
        * opaque: Towards the edges the points become more opaque and thus not visible.
        * '#FFFFFF': Towards the edges it smooths into this color.
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

    Examples
    --------
    >>> import colourmap as cm
    >>> # Compute colors per class
    >>> y=[1,1,2,2,3,1,2,3]
    >>> rgb, colordict = cm.fromlist(y)

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
    # Create opaque levels
    opaque = np.array([0.0] * len(y))

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
                opaque[Iloc] = c_gradient['opaque']
            else:
                # Set the hex colors
                c_gradient = linear_gradient(colordict.get(uiy[i]), finish_hex=gradient, n=sum(Iloc))
                colors[Iloc] = np.array(c_gradient['hex'])
                opaque[Iloc] = c_gradient['opaque']
    else:
        # Get colors for y
        colors = list(map(colordict.get, y))

        # Stack list of arrays into single array
        if scheme=='rgb':
            colors = np.vstack(colors)
        else:
            colors = np.array(colors)

    # Add a 4th column with the transparancy level.
    if len(opaque)==colors.shape[0]:
        colors = np.c_[colors, opaque]

    # Add gradient for each class
    if (X is not None) and X.shape[0]==len(y):
        if verbose>=4: print('[colourmap] >The weight of the color will be determined using the the density in input data X.')
        colors = gradient_on_density_color(X, colors, y, method='density')

    # Return
    return colors, colordict


# %%
def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    """Return a gradient list of (n) colors between two hex colors.

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

    Examples
    --------
    >>> import colourmap as cm
    >>> # Compute linear gradient for 10 points between black and white
    >>> colors = cm.linear_gradient("#000000", finish_hex="#FFFFFF", n=10)

    """
    if finish_hex=='opaque': finish_hex=start_hex
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
    coldict['opaque'] = _incremental_steps(1, 0, len(RGB_list))
    # return
    return coldict


def _incremental_steps(start, end, steps, stepsize=None):
    """Create a linear gradient between two values.

    Args:
        start (float): Starting value of the gradient.
        end (float): Ending value of the gradient.
        steps (int): Number of steps in the gradient.

    Returns:
        list: List of values representing the linear gradient.
    """
    if stepsize is None: step_size = (end - start) / (steps - 1)
    gradient = []
    for i in range(steps):
        value = start + step_size * i
        gradient.append(value)

    return gradient[0:steps]


def _color_dict(gradient):
    """Color to dictionary.

    Takes in a list of RGB sub-lists and returns dictionary of colors in RGB and
    hex form for use in a graphing function defined later on.

    """
    hex_colors = [_rgb2hex(RGB) for RGB in gradient]
    rgb_colors = np.c_[[RGB[0] for RGB in gradient], [RGB[1] for RGB in gradient], [RGB[2] for RGB in gradient]]
    return {'hex': hex_colors, 'rgb': rgb_colors}


def is_hex_color(color, verbose=3):
    """Check whether the input is a valid hex color code."""
    if not isinstance(color, str):
        if verbose>=3: print('[colourmap]> Hex [%s] should be of type string' %(str(color)))

        return False

    if color.startswith('#'):
        color = color[1:]
    else:
        if verbose>=3: print('[colourmap]> Hex [%s] should start with "#"' %(str(color)))
        return False

    if len(color) != 6:
        if verbose>=3: print('[colourmap]> Hex [%s] should be of length 7 incl "#"' %(str(color)))
        return False

    try:
        int(color, 16)
        return True
    except ValueError:
        return False


# %% Create gradient based on density.
def gradient_on_density_color(X, c_rgb, labels, method='density', showfig=False):
    """Set gradient on density color."""
    if labels is None: labels = np.repeat(0, X.shape[0])
    from scipy.stats import gaussian_kde
    uilabels = np.unique(labels)
    # Add the transparancy column of not exists
    if c_rgb.shape[1]<=3: c_rgb = np.c_[c_rgb, np.ones(c_rgb.shape[0])]
    density_colors = np.ones_like(c_rgb)

    if (len(uilabels)!=len(labels)):
        for label in uilabels:
            idx = np.where(labels==label)[0]
            if X.shape[1]==2:
                xy = np.vstack([X[idx, 0], X[idx, 1]])
            else:
                xy = np.vstack([X[idx, 0], X[idx, 1], X[idx, 2]])

            try:
                # Compute density
                z = gaussian_kde(xy)(xy)
                # Sort on density
                didx = idx[np.argsort(z)[::-1]]
                weights = _normalize(z[np.argsort(z)[::-1]])
            except:
                didx=idx

            # order colors correctly based Density
            density_colors[didx] = c_rgb[idx, :]
            # Update the transparancy level based on the density weights.
            if method=='density':
                density_colors[didx, 3]=weights

            if showfig:
                plt.figure()
                fig, ax = plt.subplots(1,2, figsize=(20,10))
                ax[0].scatter(X[didx,0], X[didx,1], color=c_rgb[idx, 0:3], alpha=c_rgb[idx, 3], edgecolor='#000000')
                ax[1].scatter(idx, idx, color=c_rgb[idx, 0:3], alpha=c_rgb[idx, 3], edgecolor='#000000')

        c_rgb=density_colors

    # Return
    return c_rgb


def _normalize(X):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    return (X - x_min) / (x_max - x_min)
