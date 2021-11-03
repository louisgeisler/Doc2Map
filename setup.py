import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    install_requires = fh.readlines()

setuptools.setup(
    name="doc2map", 
    version="1.0.0",
    author="Louis Geisler",
    description="Beautiful and interactive visualisations for NLP Topics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5', 
    py_modules=["Doc2Map"],
    package_dir={'':r'Doc2Map\src'},
    install_requires=install_requires,
    #package_data={'': [r'\doc2map\src\Doc2Map.html']},
    #setup_requires=['setuptools_scm'],
    package_data={'doc2map': ['*.html', 'src/*.html']},
    include_package_data=True,
    #zip_safe=True,
)