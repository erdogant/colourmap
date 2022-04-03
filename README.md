# colourmap

[![Python](https://img.shields.io/pypi/pyversions/colourmap)](https://img.shields.io/pypi/pyversions/colourmap)
[![PyPI Version](https://img.shields.io/pypi/v/colourmap)](https://pypi.org/project/colourmap/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/colourmap/blob/master/LICENSE)
[![Github Forks](https://img.shields.io/github/forks/erdogant/colourmap.svg)](https://github.com/erdogant/colourmap/network)
[![GitHub Open Issues](https://img.shields.io/github/issues/erdogant/colourmap.svg)](https://github.com/erdogant/colourmap/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Downloads](https://pepy.tech/badge/colourmap/month)](https://pepy.tech/project/colourmap/month)
[![Downloads](https://pepy.tech/badge/colourmap)](https://pepy.tech/project/colourmap)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/colourmap/)
<!---[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)-->
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

*`` Colourmap`` generates an unique lit of RGB and HEX colors for the specified input list.

# 
**⭐️ Star this repo if you like it ⭐️**
# 


### [Documentation pages](https://erdogant.github.io/colourmap/)

On the [documentation pages](https://erdogant.github.io/colourmap/) you can find more information about ``colourmap`` with examples. 

# 

##### Install colourmap from PyPI
```bash
pip install colourmap     # normal install
pip install -U colourmap  # update if needed
```


### Import colourmap package
```python
from colourmap import colourmap
```

<hr>

#### Quick example

Use the documentation pages for more detailed usage. Some of the most used functionalities are linked below.


```python

from colourmap import colourmap

# Create N colors
c = colourmap.generate(10)

# From list
c_rgb, c_dict = colourmap.fromlist([1,1,2,2,3,1,2,3])

# to HEX
c_hex = colourmap.rgb2hex(c_rgb)

```

#

#### [Example: Generate N unique colors from a specific colormap](https://erdogant.github.io/colourmap/pages/html/Examples.html#)

#

#### [Example: Generate unique colors based on input labels](https://erdogant.github.io/colourmap/pages/html/Examples.html#create-color-based-on-input-labels)

#

#### [Example: Create unique colors based on seaborn or matplotlib](https://erdogant.github.io/colourmap/pages/html/Examples.html#color-generated-by-seaborn-and-matplotlib)

#

#### [Example: Conversion RGB to HEX](https://erdogant.github.io/colourmap/pages/html/Examples.html#convert-rgb-to-hex)

#

#### [Example: Conversion HEX to RGB](https://erdogant.github.io/colourmap/pages/html/Examples.html#convert-rgb-to-hex)

#

#### [Example: Create a linear gradient between colors](https://erdogant.github.io/colourmap/pages/html/Examples.html#linear-gradient-between-two-colors)

<hr>

### Maintainer
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* For citations, please use the citation at the right side panel.
* If you wish to buy me a <a href="https://www.buymeacoffee.com/erdogant">Coffee</a> for this work, it is very appreciated :)
