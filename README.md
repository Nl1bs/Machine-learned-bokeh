# Machine-learned-bokeh
High quality depth of field effects are often hard to run in realtime, usually having some trick like using a lower resolution or a seperable bokeh shape to get it up to speed. If you want a gaussian or box filter it's easy to seperate into horizontal and vertical passes, but with circular bokeh (and many other shapes) it's not so easy. Bart Wronski had the idea of [using SVD to split a circle into a sequence of horizontal and vertical passes](https://bartwronski.com/2020/02/03/separate-your-filters-svd-and-low-rank-approximation-of-image-filters/), and then [use gradient descent to remove negative componants](https://bartwronski.com/2020/03/15/using-jax-numpy-and-optimization-techniques-to-improve-separable-image-filters/). This made me wonder, why not just use gradient descent for optimising two spatial passes?

This project uses machine learning to seperate circle bokeh shape into two spatial passes, and with miner mods (removing the symmetry loss and replacing the shape code) any 2d filter can be approximated with fewer samples!

Also, here's [a shadertoy demo](https://www.shadertoy.com/view/3tGcWt)
