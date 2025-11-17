import setuptools

setuptools.setup(
    name='kloom_litellm',
    version='0.1',
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'litellm',
    ]
)
