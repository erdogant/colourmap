import colourmap
print(colourmap.__version__)

# %%
colors = colourmap.linear_gradient("#000000", finish_hex="#FFFFFF", n=10)

# %%
colors = colourmap.generate(10, cmap='Set1', keep_alpha=False, verbose='info')

# %%
colors = colourmap.generate(10, cmap='Set1', keep_alpha=False, verbose=0)
colors = colourmap.generate(10, cmap='Set1', keep_alpha=False, verbose=1)
colors = colourmap.generate(10, cmap='Set1', keep_alpha=False, verbose=2)
colors = colourmap.generate(10, cmap='Set1', keep_alpha=False, verbose=3)
colors = colourmap.generate(10, cmap='Set1', keep_alpha=False, verbose=4)
colors = colourmap.generate(10, cmap='Set1', keep_alpha=False, verbose=5)

# %% Convert RGB to hex
hexcolors = colourmap.rgb2hex(colors)

# %% Convert hex to RGB
colors_2 = colourmap.hex2rgb(hexcolors)

# %%
N=10
# Create N colors
getcolors=colourmap.generate(N)
# With different cmap
getcolors=colourmap.generate(N, cmap='Set2')

# Create color for label
y=[1,1,2,2,3,1,2,3]
label_colors, colordict=colourmap.fromlist(y)

# Create color for label with gradient per group
y=[1,1,2,2,3,1,2,3]
label_colors, colordict = colourmap.fromlist(y, scheme='hex')

# With gradient
label_colors, colordict = colourmap.fromlist(y, gradient='#000000')

# With different cmap
label_colors, colordict=colourmap.fromlist(y, cmap='Set2')
# With different method
label_colors, colordict=colourmap.fromlist(y, cmap='Set2', method='seaborn')

y=['1','1','2','2','3','1','2','3']
label_colors, colordict=colourmap.fromlist(y)
# With different cmap
label_colors, colordict=colourmap.fromlist(y, cmap='Set2')
# With different method
label_colors, colordict=colourmap.fromlist(y, cmap='Set2', method='seaborn')

