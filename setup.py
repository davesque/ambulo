#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import (
    setup,
    find_packages,
)

extras_require = {
    'test': [
        "pytest==3.6",
        "pytest-cov==2.4.0",
        "tox>=2.9.1,<3",
    ],
    'lint': [
        "flake8==3.4.1",
        "isort>=4.2.15,<5",
        "pydocstyle>=3.0.0,<4",
        "mypy==0.670",
    ],
    'doc': [
        "Sphinx>=1.6.5,<2",
        "sphinx_rtd_theme>=0.1.9",
    ],
    'dev': [
        "bumpversion>=0.5.3,<1",
        "wheel",
        "twine",
        "ipython",
    ],
}

extras_require['dev'] = (
    extras_require['dev'] +
    extras_require['test'] +
    extras_require['lint'] +
    extras_require['doc']
)

setup(
    name='ambulo',
    # *IMPORTANT*: Don't manually change the version here. Use `make bump`, as described in readme
    version='0.1.0-alpha.0',
    description="""ambulo: Automatic Differentiation with Python""",
    long_description_markdown_filename='README.md',
    author='David Sanders',
    author_email='davesque@gmail.com',
    url='https://github.com/davesque/ambulo',
    include_package_data=True,
    setup_requires=['setuptools-markdown'],
    python_requires='>=3.6, <4',
    extras_require=extras_require,
    py_modules=['ambulo'],
    license="MIT",
    zip_safe=False,
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
