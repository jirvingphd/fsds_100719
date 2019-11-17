#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy','missingno', 'pandas', 'seaborn', 'matplotlib', 'scikit-learn', 
'scipy','IPython','ipywidgets','pprint','tzlocal','pyperclip','shap'] 
#'pytz','tzlocal','gensim','openpyxl','beautifulsoup4',
setup_requirements = [ 'IPython','missingno']

test_requirements = ['IPython' ,'ipywidgets']
test_requirements.extend(requirements)

setup(
    author="James Irving",
    author_email='james.irving.phd@outlook.com',
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Tools for Flatiron 100719 cohorts",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='fsds_100719',
    name='fsds_100719',
    packages=find_packages(include=['fsds_100719', 'fsds_100719.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/jirvingphd/fsds_100719',
    version='0.4.30',
    zip_safe=False,
)
