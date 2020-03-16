import colourmap
import numpy as np

def test_colourmap():
    N=np.random.randint(10)
    # Check size 1
    assert colourmap.generate(N).shape==(N,3)
    # Check size 2
    assert colourmap.generate(N, keep_alpha=True).shape==(N,4)
    # Check 3 conversion
    colors=colourmap.generate(5, cmap='Set1')
    assert np.all(colourmap.rgb2hex(colors)==['#e41a1c', '#4daf4a', '#ff7f00', '#a65628', '#999999'])
    # Check 4 conversion
    assert np.all(colourmap.hex2rgb(['#e41a1c', '#4daf4a', '#ff7f00', '#a65628', '#999999'])==colors)
    # Check alpha
    colors=colourmap.generate(3, keep_alpha=True)
    hexcolors=colourmap.rgb2hex(colors, keep_alpha=True)
    assert list(map(len,hexcolors))==[9,9,9]
    # Check no alpha
    hexcolors=colourmap.rgb2hex(colors, keep_alpha=False)
    assert list(map(len,hexcolors))==[7,7,7]
    # Check 5: fromlist
    y = ['one','two','three','one']
    out = colourmap.fromlist(y)
    assert np.all(np.isin([*out[1].keys()], y))
    # Check 6: fromlist    
    y=[1,1,2,2,3]
    out = colourmap.fromlist(y)
    assert out[0].shape==(len(y),3)
    assert len(out[1].values())==len(np.unique(y))
    


