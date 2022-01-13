# cpr-test

## General setup considerations.

I tested this env with python 3.8 and Conda (I don't usually use conda for prod, 
but I use it for my notebook environments and a lot of the work here is in notebooks).
Simply pip installing will probably work for most of the dependencies.

To install using conda:

``conda env create -f environment.yml
&& pip install -e``

- Most of the code should compile from the above, but I experimented a little with transfer learning at the end of the notebook which required a TF GPU environment. I didn't have time to complete this part and haven't tested locally; if you want to replicate this section please use colab or a TF instance on cloud.
- Please note that if you run my notebook in a colab environment, some of the plotly visualisations don't render correctly. Please use the local setup given by conda for a complete viewing (or a cloud jupyterlab environment with better settings than colab).

## How to Read Work
- I tried to stick within time limits and so the code isn't generally production ready (not unit tested, missing type hints, incomplete docstrings)
- Since I was broad in how I approached the project and pursuing numerous approaches (simple old fashioned modelling and an attempt at transfer learning), none of my approaches are complete (and the latter hasn't been debugged/verified as appropriate). As such, I've left extensive commentary in the notebook about how I would iterate on the solutions (some of the work is a sketch). Please pay close attention to the markdown comments, where I've tried to display cognizance of what could be added and what I'd do given much more time.
- Please see docstring comments too. Not production ready docstrings but some of them include important context.
- I wasn't sure if I was expected to actually create a deployment (I didn't have time anyway). But happy to discuss this and the constraints involved. I added a few notes.
- Obviously this code isn't production ready! But please email/ask if you have any questions about how I'd make it so!

Thanks and enjoy reading!

