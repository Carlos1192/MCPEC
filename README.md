MCPEC - MultiChanel data Plotting, Evaluation and Comparison.
A module for handling high-throughput multichannel data.

This module offers a collection of functions designed to evaluate
complex data sets acquired from bioreactor systems. The data collected
by these systems often contains measurements made in multiple channels
e.g. pH, dO2 levels, CO2 concentration, biomass, etc. making its
handling difficult and tedious. In many cases several cultures are
cultivated in parallel increasing the complexity of the data. The
evaluation of these measurements using spreadsheet software is
susceptible to mistakes and can take days. The functions offered in
this module aim to simplify this process as much as possible. The data
can be sorted and visualized for a qualitative graphical evaluation
within minutes. Additionally, mathematical models can be fitted to the
data to quantify the curves allowing to automate the entire evaluation
process.
For quick results check the following functions:

    load_data(filename, delimiter=',', decimal='.', sort='c'):
        Load and sort multichannel data from a CSV file.

    multiple_axes_plot(xdata, ydata, limits=[], labels=[], colors=[],
                           xlabel='', title='', off=0.15, rows=1,
                           columns=1, index=1):
        Plot multichannel data using multiple vertical axes.

    plot_image_table(xdata, ydata, rows, columns, limits=[], labels=[],
                         colors=[], xlabel='', titles=[], off=0.15,
                         fwidth=4.8, flength=6.4, padx=0.8, pady=0.2):
        Plot an image table.

    fit_sig_function(x, y, tolerance=0.1, model='sigmoid'):
        Fit a logistic sigmoid, Boltzmann or Gompertz model to the
        given data.

    fit_dsig_function(x, y, tolerance=0.1, model='sigmoid'):
        Fit a logistic double-sigmoid, double-Boltzmann or
        double-Gompertz model to the given data.

    analyze_sig_function(x, y, pars, model='sigmoid'):
        Perform a curve analysis on a sigmoid model.

    analyze_dsig_function(x, y, pars, model='sigmoid'):
        Perform a curve analysis on a sigmoid model.
