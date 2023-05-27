from colourmap.colourmap import (
    generate,
	fromlist,
    linear_gradient,
    rgb2hex,
    hex2rgb,
    hex2rgba,
    is_hex_color,
)

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.1.13'


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
