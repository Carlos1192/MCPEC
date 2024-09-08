"""Graphical evaluation of multichannel data using mcpec"""


# import modules
import mcpec as mc
from matplotlib.pyplot import savefig, close


# import csv file
x, sam, chan, cat = mc.load_data('raw_data.csv',
                                 sep=';', decimal=',', sort='s')

samples = [i[ [0, 1, 2, 3, 4] ] for i in cat]

# axes limits
limits = (
          (0, 1200),
          (0, 1200),
          (0, 1200),
          (5, 9),
          (0, 120),
          )


# axes labels
labels = (
          'Light scatter (gain 25) [u]',
          'Light scatter (gain 20) [u]',
          'Light scatter (gain 15) [u]',
          'pH',
          'dO2 [%]',
          )


# colors [R, G, B, A] (Fraunhofer colors)
colors = (
          ( 23, 156, 125),  # green
          (  0,  91, 127),  # blue
          (166, 187, 200),  # gray
          (187,   0,  86),  # red
          (  0, 133, 152),  # dark blue
          ( 57, 193, 205),  # turquoise
          (178, 210,  53),  # light green
          (245, 130,  32),  # orange
          (  0,   0,   0),  # black
          )
colors = [[i/255 for i in c] for c in colors]


# plot raw data
fig, ax = mc.plot_image_table(x, cat, rows=6, columns=8,
                              titles=sam, xlabel='Time [h]',
                              limits=limits, labels=labels,
                              colors=colors, fwidth=4.5, flength=9)
savefig('multichannel_plots.pdf', transparent=True, bbox_inches='tight')
close(fig)



# plot single plots
# ax = mc.multiple_axes_plot(x, cat[24], xlabel='Time [h]',
#                             labels=labels, colors=colors, limits=limits)
# savefig('single_plot.svg', transparent=True, bbox_inches='tight')

