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
import logging

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='[{asctime}] [{name}] [{levelname}] {msg}', style='{', datefmt='%d-%m-%Y %H:%M:%S')


# %% Main
def generate(N, cmap='Set1', method='matplotlib', keep_alpha=False, scheme='rgb', verbose='info'):
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
    verbose : int, (default: 'info')
        Print progress to screen. The default is 'info'
        60: None, 40: error, 30: warning, 20: info, 10: debug

    References
    ----------
    Colormap: https://matplotlib.org/examples/color/colormaps_reference.html

    Returns
    -------
    color_list : numpy array with colors that range between [0-1, 0-1, 0-1].

    """
    # Set the logger
    set_logger(verbose=verbose)
    listlen = 4 if keep_alpha else 3

    if method=='seaborn':
        sns = _check_seaborn()
        color_list = sns.color_palette(cmap, N)
    else:
        try:
            base = matplotlib.colormaps[cmap]
        except:
            base = plt.get_cmap(cmap)
        color_list = base(np.linspace(0, 1, N))[:, 0:listlen].tolist()
        # If there are not enough colors in the cmap, use the seaborn method.
        uicolors = len(np.unique(rgb2hex(color_list)))
        if uicolors != N:
            logger.warning('Colormap [%s] can not create [%d] unique colors! Available unique colors: [%d].' %(cmap, N, uicolors))

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
def fromlist(y, X=None, cmap='Set1', gradient=None, method='matplotlib', scheme='rgb', opaque_type='per_class', verbose='info'):
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
        * opaque: Towards the edges the points become more opaque and thus not visible. Note that scheme must be 'rgb'
        * '#FFFFFF': Towards the edges it smooths into this color.
    method : String, optional
        Method to generate colors
        'matplotlib' (default)
        'seaborn'
    scheme : String, optional
        The output of color is in the scheme:
        'rgb'
        'hex'
    opaque_type : String, optional
        Method to generate transparency. Requires scheme='rgb' and input data X
            * 'per_class': Transprancy is determined on the density within the class label (y)
            * 'all': Transprancy is determined on all available data points
            * 'lineair': Transprancy is lineair set within the class label (y)
    verbose : int, (default: 'info')
        Print progress to screen. The default is 'info'
        60: None, 40: error, 30: warning, 20: info, 10: debug

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
    # Set the logger
    set_logger(verbose=verbose)
    # make unique
    y = np.array(y)
    uiy = np.unique(y)

    # Get colors
    colors_unique = generate(len(uiy), cmap=cmap, method=method, scheme=scheme, verbose=verbose)

    # Make dict for each search
    colordict = dict(zip(uiy, colors_unique))
    # Create opaque levels
    opaque = np.array([1.0] * len(y))

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

    # Add a 4th column with the transparency level.
    if scheme=='rgb':
        logger.info('Add transparency to RGB colors (last column)')
        colors = np.c_[colors, opaque]

        # Add gradient for each class
        if (X is not None) and X.shape[0]==len(y):
            colors = gradient_on_density_color(X, colors, y, opaque_type=opaque_type, verbose=verbose)

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
    if stepsize is None: step_size = (end - start) / np.maximum((steps - 1), 1)
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


def is_hex_color(color, verbose='info'):
    """Check whether the input is a valid hex color code.

    verbose : int, (default: 'info')
        Print progress to screen. The default is 'info'
        60: None, 40: error, 30: warning, 20: info, 10: debug
    """
    # Set the logger
    set_logger(verbose=verbose)

    if not isinstance(color, str):
        logger.info('Hex [%s] should be of type string' %(str(color)))

        return False

    if color.startswith('#'):
        color = color[1:]
    else:
        logger.info('Hex [%s] should start with "#"' %(str(color)))
        return False

    if len(color) != 6:
        logger.info('Hex [%s] should be of length 7 incl "#"' %(str(color)))
        return False

    try:
        int(color, 16)
        return True
    except ValueError:
        return False


# %% Create gradient based on density.
def gradient_on_density_color(X, c_rgb, labels, opaque_type='per_class', showfig=False, verbose='info'):
    """Set gradient on density color.

    This function determines the density of the data and adds a transparency column.
    If samples are in dense areas, transparency values are towards 1 (visible), whereas isn none-dense areas,
    the transparency values are towards 0 (not visible).

    Parameters
    ----------
    X : Array-like
        Input data to determine the density.
    c_rgb : array-like of type Nx3 or Nx4
        RGB colors.
    labels: list of labels with same size as X
        labels of the samples.
    opaque_type : String, optional
            * 'per_class': Transprancy is determined on the density within the class label (y)
            * 'all': Transprancy is determined on all available data points
            * 'lineair': Transprancy is lineair set within the class label (y)
    showfig : Bool, default: False
        Show figure as sanity check.
    verbose : int, (default: 'info')
        Print progress to screen. The default is 'info'
        60: None, 40: error, 30: warning, 20: info, 10: debug

    Returns
    -------
    c_rgb : array-like of Nx4
        RGB for which the last column is the transparency.

    """
    # Set the logger
    set_logger(verbose=verbose)

    if labels is None: labels = np.repeat(0, X.shape[0])
    from scipy.stats import gaussian_kde
    uilabels = np.unique(labels)
    # Add the transparency column of not exists
    if c_rgb.shape[1]<=3: c_rgb = np.c_[c_rgb, np.ones(c_rgb.shape[0])]
    density_colors = np.ones_like(c_rgb)

    if opaque_type=='all':
        try:
            # Compute density
            z = gaussian_kde(X.T)(X.T)
            weights = _normalize(z[np.argsort(z)[::-1]])
            c_rgb[:, 3] = weights
        except:
            pass

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
                passed = True
                # weights = _normalize(z[np.argsort(z)[::-1]])
            except:
                didx=idx
                passed = False

            # order colors correctly based Density
            density_colors[didx] = c_rgb[idx, :]

            # Update the transparency level based on the density weights.
            if opaque_type=='per_class':
                weights = _normalize(z[np.argsort(z)[::-1]]) if passed else np.ones_like(idx)
                density_colors[didx, 3] = weights

            if showfig:
                plt.figure()
                fig, ax = plt.subplots(1, 2, figsize=(20, 10))
                ax[0].scatter(X[didx, 0], X[didx, 1], color=c_rgb[idx, 0:3], alpha=c_rgb[idx, 3], edgecolor='#000000')
                ax[1].scatter(idx, idx, color=c_rgb[idx, 0:3], alpha=c_rgb[idx, 3], edgecolor='#000000')

        c_rgb=density_colors

    # Return
    return c_rgb


def _normalize(X):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    out = (X - x_min) / (x_max - x_min)
    out[np.isnan(out)]=1
    return out


#%%
def convert_verbose_to_new(verbose):
    """Convert old verbosity to the new one."""
    # In case the new verbosity is used, convert to the old one.
    if verbose is None: verbose=0
    if not isinstance(verbose, str) and verbose<10:
        status_map = {
            'None': 'silent',
            0: 'silent',
            6: 'silent',
            1: 'critical',
            2: 'warning',
            3: 'info',
            4: 'debug',
            5: 'debug'}
        if verbose>=2: print('[colourmap] WARNING use the new verbose status. This will be deprecated in future versions.')
        return status_map.get(verbose, 0)
    else:
        return verbose


def get_logger():
    """Return logger status."""
    return logger.getEffectiveLevel()


def set_logger(verbose: [str, int] = 'info', return_status: bool = False):
    """Set the logger for verbosity messages.

    Parameters
    ----------
    verbose : str or int, optional, default='info' (20)
        Logging verbosity level. Possible values:
        - 0, 60, None, 'silent', 'off', 'no' : no messages.
        - 10, 'debug' : debug level and above.
        - 20, 'info' : info level and above.
        - 30, 'warning' : warning level and above.
        - 50, 'critical' : critical level and above.

    Returns
    -------
    None.

    > # Set the logger to warning
    > set_logger(verbose='warning')
    > # Test with different messages
    > logger.debug("Hello debug")
    > logger.info("Hello info")
    > logger.warning("Hello warning")
    > logger.critical("Hello critical")

    """
    # Convert verbose to new
    verbose = convert_verbose_to_new(verbose)
    # Set 0 and None as no messages.
    if (verbose==0) or (verbose is None):
        verbose=60
    # Convert str to levels
    if isinstance(verbose, str):
        levels = {
            'silent': logging.CRITICAL + 10,
            'off': logging.CRITICAL + 10,
            'no': logging.CRITICAL + 10,
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL,
        }
        verbose = levels[verbose]

    # Show examples
    logger.setLevel(verbose)

    if return_status:
        return verbose


def check_logger(verbose: [str, int] = 'info'):
    """Check the logger."""
    set_logger(verbose)
    logger.debug('DEBUG')
    logger.info('INFO')
    logger.warning('WARNING')
    logger.critical('CRITICAL')


def disable_tqdm():
    """Set the logger for verbosity messages."""
    return (True if (logger.getEffectiveLevel()>=30) else False)
