from colourmap.colourmap import (
    generate,
	fromlist,
    rgb2hex,
    hex2rgb,
)

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.0'


# module level doc-string
__doc__ = """
**colourmap** 
=====================================================================

Description
-----------
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
