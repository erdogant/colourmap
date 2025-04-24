import logging

from colourmap.colourmap import (
    generate,
	fromlist,
    linear_gradient,
    rgb2hex,
    hex2rgb,
    hex2rgba,
    is_hex_color,
    gradient_on_density_color,
    check_logger,
)

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.1.21'

# Setup root logger
_logger = logging.getLogger('colourmap')
_log_handler = logging.StreamHandler()
_fmt = '[{asctime}] [{name}] [{levelname}] {msg}'
_formatter = logging.Formatter(fmt=_fmt, style='{', datefmt='%d-%m-%Y %H:%M:%S')
_log_handler.setFormatter(_formatter)
_log_handler.setLevel(logging.DEBUG)
_logger.addHandler(_log_handler)
_logger.propagate = False

# module level doc-string
__doc__ = """
**colourmap**
=====================================================================

colourmap generates an N unique colors from the specified input colormap.

Examples
--------
>>> import colourmap
>>> N=5
>>> colors = colourmap.generate(N)
>>> hexcolors = colourmap.rgb2hex(colors)
>>> colors = colourmap.hex2rgb(hexcolors)
>>> colors = colourmap.fromlist([1,1,2,2,3])

References
----------
* https://github.com/erdogant/colourmap

"""
