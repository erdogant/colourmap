import setuptools
#import versioneer
new_version='0.1.0'

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     install_requires=['matplotlib','numpy'],
     python_requires='>=3',
     name='colourmap',
     version=new_version,
#     version=versioneer.get_version(),    # VERSION CONTROL
#     cmdclass=versioneer.get_cmdclass(),  # VERSION CONTROL
     author="Erdogan Taskesen",
     author_email="erdogant@gmail.com",
     description="Python package colourmap generates an N unique colors from the specified input colormap.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/erdogant/colourmap",
	 download_url = 'https://github.com/erdogant/colourmap/archive/'+new_version+'.tar.gz',
     packages=setuptools.find_packages(), # Searches throughout all dirs for files to include
     include_package_data=True, # Must be true to include files depicted in MANIFEST.in
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
