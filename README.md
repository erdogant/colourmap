# colourmap

[![Python](https://img.shields.io/pypi/pyversions/colourmap)](https://img.shields.io/pypi/pyversions/colourmap)
[![PyPI Version](https://img.shields.io/pypi/v/colourmap)](https://pypi.org/project/colourmap/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/colourmap/blob/master/LICENSE)
[![Github Forks](https://img.shields.io/github/forks/erdogant/colourmap.svg)](https://github.com/erdogant/colourmap/network)
[![GitHub Open Issues](https://img.shields.io/github/issues/erdogant/colourmap.svg)](https://github.com/erdogant/colourmap/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Downloads](https://pepy.tech/badge/colourmap/month)](https://pepy.tech/project/colourmap/month)
[![Downloads](https://pepy.tech/badge/colourmap)](https://pepy.tech/project/colourmap)
<!---[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)-->
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

* Python package colourmap generates an N unique colors from the specified input colormap.

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

#

#### [Example: Check whether the elements of X are present in Y](https://erdogant.github.io/ismember/pages/html/Examples.html#)

#

#### [Example: Determine the corresponding location of the values that are present in Y array](https://erdogant.github.io/ismember/pages/html/Examples.html#determine-the-corresponding-location-of-the-values-that-are-present-in-y-array)

#

#### [Example: Row wise comparison](https://erdogant.github.io/ismember/pages/html/Examples.html#row-wise-comparison-1)

#

#### [Example: Elementwise comparison](https://erdogant.github.io/ismember/pages/html/Examples.html#elementwise-comparison)


#### Quick examples


### Example:
```python
N=10
# Create N colors
getcolors=colourmap.generate(N)

# With different cmap
getcolors=colourmap.generate(N, cmap='Set2')

# Create color for label
y=[1,1,2,2,3,1,2,3]
label_colors, colordict=colourmap.fromlist(y)
# With different cmap
label_colors, colordict=colourmap.fromlist(y, cmap='Set2')
# With different method
label_colors, colordict=colourmap.fromlist(y, cmap='Set2', method='seaborn')

# String as input labels
y=['1','1','2','2','3','1','2','3']
label_colors, colordict=colourmap.fromlist(y)
# With different cmap
label_colors, colordict=colourmap.fromlist(y, cmap='Set2')
# With different method
label_colors, colordict=colourmap.fromlist(y, cmap='Set2', method='seaborn')

```

### Citation
Please cite colourmap in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdogant2019colourmap,
  title={colourmap},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/colourmap}},
}
```

### Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

### Contribute
* Contributions are welcome.

### Licence
See [LICENSE](LICENSE) for details.

### Donation
* This work is created and maintained in my free time. If you wish to buy me a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a> for this work, it is very appreciated.
