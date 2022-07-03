# cpr-test

## General setup considerations.

I tested this env with python 3.8 and Conda (I don't usually use conda for prod, 
but I use it for my notebook environments and a lot of the work here is in notebooks).
Simply pip installing will probably work for most of the dependencies.

To install using conda:

``conda env create -f environment.yml
&& pip install -e``

- Most of the code should compile from the above, but I experimented a little with transfer learning at the end of the notebook which required a TF GPU environment. I didn't have time to pursue this approach properly and so got bugs/non-useful results. In hindsight I'd have spent more time on the more fruitful and working approach but I've left the boilerplate code in for visibility; if you want to replicate the small bit I've done on this section use colab or a TF instance on cloud to avoid tf setup. Ask if needed.
- Please note that if you run my notebook in a colab environment, some of the plotly visualisations don't render correctly. Please use the local setup given by conda for a complete viewing (or a cloud jupyterlab environment with better viz settings than colab).

## How to Read Work
- The most important thing is to use the notebook as a guide for everything else. Please pay close attention to the markdown comments in the notebook. I've modularised the code base a bit for readability, but not exaclty how I'd design a prod ready package.
- I tried to stick within time limits and so some shortcuts were made (no unit tests, missing type hints, incomplete docstrings, some duplication in the notebook, some hacks). Lots more could be done, but I've included lots of comments and can field further questions about improvements.
- Since I was broad in how I approached the project and pursued numerous approaches (simple old fashioned modelling and a very quick attempt at transfer learning at the end) no approach is explored too deeply (and the transfer learning approach is buggy and only left in for visibility). I've left extensive commentary in the notebook about how I would iterate on the solutions.
- Please see docstring comments too. Not production ready docstrings but some of them include important context.
- I wasn't sure if I was expected to actually create a deployment (I didn't have time anyway). But happy to discuss this and the constraints involved. I added a few notes in the notebook.
- Please email/ask if you have any questions about how I'd make it so!

Thanks and enjoy reading!

## Installing XGBoost from source

* [Here](https://xgboost.readthedocs.io/en/stable/build.html) is a quick guide to installing XGBoost from source.
* Note, if you're using a linux machine for xgboost you may need an earlier gcc compiler. These are helpful links [here](https://askubuntu.com/questions/1039856/downgrade-gnu-compilers-ubuntu-18-04) and [here](https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-20-04-lts-focal-fossa) In general, you don't want to downgrade for the system,
so follow the links for switching.

