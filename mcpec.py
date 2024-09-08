"""MultiChannel data Plotting, Evaluation and Comparison.

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

"""


#######################################################################
#######################################################################


def insert_missing_samples(ref, sample_names, measurement, chan):
    from numpy import array

    if ref in sample_names:
        return measurement[ref == sample_names][0]
    else:
        print(f'{chan}: Sample {ref} missing. Replaced with 0s')
        return array([0]*len(measurement[0]))



def iterate_missing_samples(ref, sample_names, measurements, chan):
    from numpy import array

    return array(
        [insert_missing_samples(r, sample_names, measurements, chan)
         for r in ref]
        )



def load_data(filename, sep=';', decimal='.',
              x_identifier='TIME [h] ->', sort='c'):
    """ Load and sort multichannel data from a CSV file.

    Import a CSV file, determine samples and channels, check for missing
    measurements, sort and return a list of arrays sorted by samples or
    channels. The raw-data must contain the names of the samples in the first
    column and the names of the channels in another column below the
    identifier keyword (default: 'TIME [h] ->'). All columns between these
    two are ignored. The time is given from left to right in the same row as
    the identifier keyword. All rows above this keyword are ignored. The
    measurements are given in rows from left to right below the corresponding
    timestamp. Missing samples are replaced by rows of 0s. Single empty cells
    are interpreted as numpy.nan.

    Parameters
    ----------
    filename : str
        csv file to be imported.
    sep : str, optional
        Separator between values. The default is ';'.
    decimal : str, optional
        Character to recognize as decimal point. The default is '.'.
    x_identifier : str, optional
        Keyword to identify x values and channels. The default is
        'TIME [h] ->'.
    sort : str, optional
        Sort the data by channel or by sample. If 'c' is set, the output
        arrays are sorted by channel containing all samples of the same
        channel. Else, the output arrays are sorted by samples
        containig all channels of the same samples. The default is 'c'.

    Returns
    -------
    names : ndarray
        1D-array containing the names of the samples.
    channels : ndarray
        1D-array containing the channels of the samples.
    x_values : ndarray
        1D-array containing the x values.
    samples : [ndarray]
        List of 2D-arrays containing the sorted data. The arrays contain all
        samples of the same channel or all channels of the same sample
        (see 'sort'). The measurements are sorted by rows.

    """

    from numpy import array, where, unique, isnan

    with open(filename, 'r') as reader:
        reads = reader.readlines()
        count = max([i.count(sep) for i in reads])
        table = [i + sep*(count - i.count(sep)) for i in reads]
        table = [i.replace(decimal, '.') for i in table]
        table = [i.replace('\n', '') for i in table]
        table = array([i.split(sep) for i in table])
        non_empty_rows = (table == '').sum(axis=1) != len(table[0, :])
        table = table[non_empty_rows]

    # find coordinates of x_values
    first_row, x_col = [i[0] for i in where(table == x_identifier)]
    last_row = max(where(table[:, x_col] != '')[0]) + 1

    # x_values, sample_names, channels, y_measurements
    x_values = table[first_row,              x_col+1 : -1].astype(float)
    names    = table[first_row+1 : last_row, 0           ].astype(str)
    chans    = table[first_row+1 : last_row, x_col       ].astype(str)
    data     = table[first_row+1 : last_row, x_col+1 : -1].astype(float)

    # search for numpy.nans
    for n, c, d in zip(names, chans, data):
        if True in isnan(d):
            nan_indexes = where(isnan(d))[0]
            print(f'{c}: {n} contains numpy.nans {nan_indexes}')

    # sort data
    if sort == 'c':
        sort_nam = [names[chans == c] for c in unique(chans)]
        sort_dat = [data[chans == c] for c in unique(chans)]
        samples  = [iterate_missing_samples(unique(names), sn, sd, c)
                    for sn, sd, c in zip(sort_nam, sort_dat, unique(chans))]
    else:
        sort_nam = [chans[names == n] for n in unique(names)]
        sort_dat = [data [names == n] for n in unique(names)]
        samples  = [iterate_missing_samples(unique(chans), sn, sd, c)
                    for sn, sd, c in zip(sort_nam, sort_dat, unique(names))]

    return x_values, unique(names), unique(chans), samples


#######################################################################
#######################################################################


def remove_nans(x, y):
    """Remove numpy.nan values keeping both arrays the same size.

    Parameters
    ----------
    xdata : 1D-array
        Horizontal axis.
    ydata : 1D-array
        Vertical axis.

    Returns
    -------
    x : 1D-array
        Array without numpy.nans.
    y : 1D-array
        Array without numpy.nans.

    """

    from numpy import isnan
    return x[isnan(y) == False], y[isnan(y) == False]



def multiple_axes_plot(xdata, ydata, limits=[], labels=[], colors=[],
                       xlabel='', title='', off=0.15, rows=1, columns=1,
                       index=1):
    """Plot multichannel data using multiple vertical axes.

    Plot xdata versus ydata adding a new vertical axis to the end of the
    horizontal axis for every row in ydata. xdata and every row in ydata
    have be the same length.

    Parameters
    ----------
    xdata : 1D-array
        Horizontal axis.
    ydata : 2D-array
        vertical axes. Every row of the array will be plotted as an own
        vertical axis.
    limits : [[float, float]], optional
        Lower and upper limit of the vertical axes. If used the amount
        of elements in the list must match or exceed the number of rows
        in ydata. If not used the data is plotted from the 5th to the
        95th percentile to discard potential outliers. The default is [].
    labels : [str], optional
        Labels of the vertical axes. If used the amount of elements in the
        list must match or exceed the number of rows in ydata.
        The default is [].
    colors : [[int, int, int, int]], optional
        Colors. If used the amount of elements in the list must match
        or exceed the number of rows in ydata. The colors are
        defined as RGBA or RGB from 0 to 255. They are assigned to the
        graph, the corresponding axis and the label of the axis. The default
        is [].
    xlabel : str, optional
        Label of the horizontal axis. The default is ''.
    title : str, optional
        Text to use for the title. The default is ''.
    off : float, optional
        Offset (space) between the vertical axes. The default is 0.15.

    Returns
    -------
    bx : Axes object
        Final plot.

    """

    from numpy import percentile, isnan
    from matplotlib.pyplot import subplot

    if limits == []:
        limits = [percentile(i[~isnan(i)], (5,95)) for i in ydata]
    if labels == []:
        labels = [None]*len(ydata)
    if colors == []:
        colors = [f'C{i}' for i, y in enumerate(ydata)]

    ax = subplot(rows, columns, index, title=title, xlabel=xlabel)
    ax.set_ylim(limits[0])
    xdat, ydat = remove_nans(xdata, ydata[0])
    ax.plot(xdat, ydata[0], color=colors[0])
    ax.set_ylabel(labels[0], color=colors[0])
    ax.tick_params(axis='y', colors=colors[0])

    for i, (y, lim, lab, col) in enumerate(
            zip(ydata[1:], limits[1:], labels[1:], colors[1:])):
        bx = ax.twinx()
        bx.set_ylim(lim)
        xdat, ydat = remove_nans(xdata, y)
        bx.plot(xdat, ydat, color=col)
        bx.set_ylabel(lab, color=col)
        bx.tick_params(axis='y', colors=col)
        bx.spines['right'].set_position(('axes', 1+i*off))

    return bx



def plot_image_table(xdata, ydata, rows, columns, limits=[], labels=[],
                     colors=[], xlabel='', titles=[], off=0.15, fwidth=4.8,
                     flength=6.4, padx=0.8, pady=0.2):
    """Plot an image table.

    Plots the alements of ydata in a table defined by the number of rows and
    columns. The plots are plotted from left to right.

    Parameters
    ----------
    xdata : 1D-array
        Horizontal axis.
    ydata : [2D-array]
        vertical axes. Every element is plotted from left to right with
        every row of the array plotted as an own vertical axis.
    rows : int
        Number of rows.
    columns : int
        Number of columns.
    limits : [[float, float]], optional
        Lower and upper limit of the vertical axes. If used the amount
        of elements in the list must match or exceed the number of rows
        in ydata. If not used the data is plotted from the 5th to the
        95th percentile to discard potential outliers. The default is [].
    labels : [str], optional
        Labels of the vertical axes. If used the amount of elements in the
        list must match or exceed the number of rows in ydata.
        The default is [].
    colors : [[int, int, int, int]], optional
        Colors. If used the amount of elements in the list must match
        or exceed the number of rows in ydata. The colors are
        defined as RGBA or RGB from 0 to 255. They are assigned to the
        graph, the corresponding axis and the label of the axis. The default
        is [].
    xlabel : str, optional
            Label of the horizontal axes. The default is ''.
    titles : [str], optional
        Text to use for the titles. The default is [].
    off : float, optional
        Offset (space) between the vertical axes. The default is 0.15.
    fwidth : float, optional
        Width of the figure. The default is 4.8.
    flength : float, optional
        Length of the figure. The default is 6.4.
    padx : float, optional
        Horizontal padding (space) between the plots. The default is 0.8.
    pady : float, optional
        Vertical padding (space) between the plots. The default is 0.2.

    Returns
    -------
    fig : Figure object
        Final figure.
    ax : Axes object
        Plots.

    """

    from matplotlib.pyplot import figure, subplots_adjust

    if titles == []: titles = [None]*len(ydata)

    fig = figure(figsize=[flength*columns, fwidth*rows])

    for i, y, t in zip(range(rows*columns), ydata, titles):
        ax = multiple_axes_plot(xdata, y, limits=limits, labels=labels,
                                colors=colors, xlabel=xlabel, title=t,
                                off=off, rows=rows, columns=columns,
                                index=i+1)

        fig.add_subplot(ax)
        print(i, t)

    subplots_adjust(left=0, bottom=0, right=1, top=1,
                    wspace=padx, hspace=pady)

    return fig, ax


#######################################################################
#######################################################################


def sigmoid(x, a, b, c, d):
    # reparameterized logistic sigmoid function
    from numpy import exp as e
    return c/(1+e(a*(b-x))) + d

def sigmoid_d1(x, a, b, c):
    # simplified first derivation of the logistic sigmoid function
    sig = sigmoid(x, a, b, 1, 0)
    return sig*c * (1-sig)*a

def sigmoid_d2(x, a, b, c):
    # simplified second derivation of the logistic sigmoid function
    sig = sigmoid(x, a, b, 1, 0)
    return sigmoid_d1(x, a, b, c) * (1-2*sig)*a



def boltzmann(x, a, b, c, d):
    # Boltzmann function
    from numpy import exp as e
    return (c-d) / (1+e((b-x)/a)) + d

def boltzmann_d1(x, a, b, c, d):
    # simplified first derivation of the Boltzmann function
    sig = boltzmann(x, a, b, 1, 0)
    return sig*(c-d) * (1-sig)/a

def boltzmann_d2(x, a, b, c, d):
    # simplified second derivation of the Boltzmann function
    sig = boltzmann(x, a, b, 1, 0)
    return boltzmann_d1(x, a, b, c, d) * (1-2*sig)/a



def gompertz(x, a, b, c, d):
    # Gompertz function
    from numpy import exp as e
    return c*e(-b*e(-a*x)) + d

def gompertz_d1(x, a, b, c):
    # simplified first derivation of the Gompertz function
    from numpy import log
    sig = log(gompertz(x, a, b, 1, 0))
    return gompertz(x, a, b, c, 0) * sig * (-a)

def gompertz_d2(x, a, b, c):
    # simplified second derivation of the Gompertz function
    from numpy import log
    sig = log(gompertz(x, a, b, 1, 0))
    return gompertz_d1(x, a, b, c) * (sig*(-a)-a)



def is_array(arr):
    from numpy import ndarray
    return type(arr) is ndarray



def calc_slopes(x, y):
    if is_array(x) and is_array(y):
        slopes = (y[1:]-y[:-1]) / (x[1:]-x[:-1])
        return x[1:], slopes
    else:
        raise TypeError('ensure x and y are arrays')



def estimate_pars(x, y):
    """Estimate the starting parameters to fit the models

    Calculates the quotient of the difference of y and x to estimate the
    slopes of the data. The new curve is used to estimate the slope, the point
    of inflection (poi), the peak and the intercept.

    Parameters
    ----------
    x : 1D-array
        Values corresponding to the horizontal axis.
    y : 1D-array
        Values corresponding to the vertical axis.

    Returns
    -------
    slope : float
        Slope at the poi.
    poi : float
        Point of inflection.
    peak : float
        Peak.
    intercept : float
        Intercept.

    """

    dx, dy = calc_slopes(x, y)

    slope = max(dy)
    poi = dx[dy == slope][0]
    peak = max(y)
    intercept = y[x == min(abs(x))][0]
    return slope, poi, peak, intercept



def estimate_bounds(x, y, pars, usetol=[], tolerance=0.1):
    """ Estimate lower and upper boundary for the estimated parameters.

    Defines the lower and upper boundaries ranging from 0 to infinity. If
    "usetol" is given, the corresponding boundary is adjusted to the value of
    the given parameter +/- a deviation in x or y direction. The deviation is
    determined as the range of x or y multiplied by the tolerance.

    Parameters
    ----------
    x : 1D-array
        Values corresponding to the horizontal axis.
    y : 1D-array
        Values corresponding to the vertical axis.
    pars : [float]
        Initial guesses for the parameters.
    usetol : [int, "x"] or [int, "y"], optional
        The int defines the index of the parameter to be defined. The str
        defines the axis of the deviation to be used. The default is [].
    tolerance : float, optional
        Multiplier to adjust the deviation from pars. The default is 0.1.

    Raises
    ------
    TypeError
        Raised when the elements of "usetol" do not match the expected format.

    Returns
    -------
    lb : [float]
        Lower boundaries of "pars".
    ub : [float]
        Upper boundaries of "pars".

    """

    if is_array(x) and is_array(y):
        from numpy import inf
        # define deviation from estimated parameters
        dev_x = (max(x) - min(x))*tolerance
        dev_y = (max(y) - min(y))*tolerance
        # define lower and upper boundary for the estimated parameters
        lb =   [0]*len(pars)
        ub = [inf]*len(pars)

        for tol in usetol:
            # define if tolerance from x or y axis applies
            if tol[1] == 'x': dev = dev_x
            elif tol[1] == 'y': dev = dev_y
            else: raise TypeError(
                'usetol elements must be list-like: [int, "x"] or [int, "y"]')
            lb[tol[0]] = pars[tol[0]] - dev
            ub[tol[0]] = pars[tol[0]] + dev
        return lb, ub

    else:
        raise TypeError('ensure x and y are arrays')



def curve_fitting(x, y, func, p0, bounds):
    """Fit a model to the given data.

    Adjust the parameters of func to best fit the data using
    scipy.optimize.curve_fit(). The initial guesses p0 are crucial for
    a successful fit. bounds limits the range of the parameters to
    avoid overfitting.

    Parameters
    ----------
    xdata : 1D-array
        Values corresponding to the horizontal axis.
    ydata : 1D-array
        Values corresponding to the vertical axis.
    func : callable
        Model function to be fitted to the data.
    p0 : [float]
        Initial guesses for the parameters of func.
    bounds : [[float], [float]]
        Lower and upper bounds of the parameters of func.

    Returns
    -------
    pars : [float]
        Optimal values for the parameters so that the sum of the squared
        residuals of f(xdata, *popt) - ydata is minimized.
    pcov : [[float]]
        The estimated covariance of pars.
    r2 : float
        Coefficient of determination.
    rmsd : float
        Root-mean-square deviation.

    """
    from scipy.optimize import curve_fit
    from numpy import mean

    pars, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds)

    # calculate R2 and RMSD
    residuals = y - func(x, *pars)
    ss_res = sum(residuals**2)
    ss_tot = sum((y-mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    adj_r2 = 1-(1-r2)*((len(y)-1)/ (len(y)-len(pars)-1))
    rmsd = ss_res**0.5

    return pars, pcov, r2, adj_r2, rmsd



def fit_sig_function(x, y, tolerance=0.1, model='sigmoid'):
    """Fit a logistic sigmoid, Boltzmann or Gompertz model to the given data.

    Estimates and adjusts the parameters and boundaries for the given model
    bevor fitting.

    Parameters
    ----------
    x : 1D-array
        Values corresponding to the horizontal axis.
    y : 1D-array
        Values corresponding to the vertical axis.
    tolerance : float, optional
        Deviation from the given parameter. The default is 0.1.
    model : str, optional
        Model function to be fitted to the data. "sigmoid", "boltzmann" and
        "gompertz" are available. The default is "sigmoid".

    Raises
    ------
    ValueError
        Raised when the model is unknown.

    Returns
    -------
    pars : [float]
        Optimal values for the parameters so that the sum of the squared
        residuals of f(xdata, *popt) - ydata is minimized.
    pcov : [[float]]
        The estimated covariance of pars.
    r2 : float
        Coefficient of determination.
    rmsd : float
        Root-mean-square deviation.

    """

    if model == 'sigmoid':
        def func(x, a, b, c, d): return sigmoid(x, a, b, c, d)
        def rearange_pars(a, b, c, d): return (a, b, c-d, d)
        apply_tol = [(1, 'x'), (2, 'y'), (3, 'y')]

    elif model == 'boltzmann':
        def func(x, a, b, c, d): return boltzmann(x, a, b, c, d)
        def rearange_pars(a, b, c, d): return (a, b, c, d)
        apply_tol = [(1, 'x'), (2, 'y'), (3, 'y')]

    elif model == 'gompertz':
        def func(x, a, b, c, d): return gompertz(x, a, b, c, d)
        def rearange_pars(a, b, c, d): return (a, b, c-d, d)
        apply_tol = [(2, 'y'), (3, 'y')]

    else: raise ValueError(
        'unknown function. Choose "sigmoid", "boltzmann" or "gompertz"')


    a, b, c, d = estimate_pars(x, y)
    est_pars = rearange_pars(a, b, c, d)
    est_bounds = estimate_bounds(x, y, est_pars, usetol=apply_tol,
                                 tolerance=tolerance)

    return curve_fitting(x, y, func, est_pars, est_bounds)



def calc_tangent(x, func, func_d, poi_x, py=[]):
    """Calculate a tangent of a given function at a given point.

    Calculate a tangent for func(x) at a given point using the derivation of
    func to determine the slope. The tangent can be limited to a range
    determined by the vertical axis defined in "py".

    Parameters
    ----------
    x : 1D-array
        Values corresponding to the horizontal axis.
    func : callable
        Function to calculate the values of the vertical axis.
    func_d : callble
        First derivation of "func" required to determine the slope.
    poi_x : float
        Point to calculate the tangent.
    py : [float], optional
        Limits of the tangent in vertical direction. The default is [].

    Raises
    ------
    TypeError
        Raised if x is not an array.

    Returns
    -------
    x_tangent : 1D-array
        Values corresponding to the horizontal axis.
    y_tangent : 1D-array
        Values corresponding to the vertical axis.

    """

    if is_array(x):
        # f(x) = slope*x + intercept
        # intercept = f(x) - slope*x
        # x = (y - intercept) / slope
        slope = func_d(poi_x)
        intercept = func(poi_x) - slope*poi_x

        # limit tangent according to y-values
        if len(py) > 0:
            # define x coordinates to given y coordinates
            py_x = [(i-intercept)/slope for i in py]
            # limit x to given coordinates
            x = x[x >= min(py_x)]
            x = x[x <= max(py_x)]
        return x, slope*x+intercept

    else:
        raise TypeError('x is not an array')



def approximate_root(x, y):
    # returns the sign changing interval (x0, x1) from y
    if is_array(x) and is_array(y):
        x_root_0 = x[:-1][y[:-1]*y[1:] < 0]
        x_root_1 = x[1: ][y[:-1]*y[1:] < 0]
        return x_root_0, x_root_1
    else:
        raise TypeError('ensure x and y are arrays')



def fit_dsig_function(x, y, tolerance=0.1, model='sigmoid'):
    """Fit a logistic double-sigmoid, double-Boltzmann or double-Gompertz
    model to the given data.

    Estimates and adjusts the parameters and boundaries for the given model
    bevor fitting.

    Parameters
    ----------
    x : 1D-array
        Values corresponding to the horizontal axis.
    y : 1D-array
        Values corresponding to the vertical axis.
    tolerance : float, optional
        Deviation from the given parameter. The default is 0.1.
    model : str, optional
        Model function to be fitted to the data. "sigmoid", "boltzmann" and
        "gompertz" are available. The default is "sigmoid".

    Raises
    ------
    ValueError
        Raised when the model is unknown.

    Returns
    -------
    pars : [float]
        Optimal values for the parameters so that the sum of the squared
        residuals of f(xdata, *popt) - ydata is minimized.
    pcov : [[float]]
        The estimated covariance of pars.
    r2 : float
        Coefficient of determination.
    rmsd : float
        Root-mean-square deviation.

    """

    if model == 'sigmoid':
        def func(x, a, b, c, d, a2, b2, c2):
            return sigmoid(x, a, b, c, d) - sigmoid(x, a2, b2, c2, 0)
        def rearange_pars(a, b, c, d, a2, b2, c2, d2):
            return (a, b, c-d, d, a2, b2, c+c2)
        apply_tol = [(1, 'x'), (2, 'y'), (3, 'y'), (5, 'x'), (6, 'y')]

    elif model == 'boltzmann':
        def func(x, a, b, c, d, a2, b2, d2):
            return boltzmann(x, a, b, c, d) - boltzmann(x, a2, b2, c, d2) + d2
        def rearange_pars(a, b, c, d, a2, b2, c2, d2):
            return (a, b, c, d, a2, b2, -d2)
        apply_tol = [(1, 'x'), (2, 'y'), (3, 'y'), (5, 'x'), (6, 'y')]

    elif model == 'gompertz':
        def func(x, a, b, c, d, a2, b2, c2):
            return gompertz(x, a, b, c, d) - gompertz(x, a2, b2, c2, 0)
        def rearange_pars(a, b, c, d, a2, b2, c2, d2):
            return (a, b, c-d, d, a2, b2, c+c2)
        apply_tol = [(2, 'y'), (3, 'y'), (6, 'y')]

    else: raise ValueError(
        'unknown function. Choose "sigmoid", "boltzmann" or "gompertz"')


    peak_x = x[y == max(y)]
    a,  b,  c,  d  = estimate_pars(x[x <= peak_x],  y[x <= peak_x])
    a2, b2, d2, c2 = estimate_pars(x[x >= peak_x], -y[x >= peak_x])
    est_pars = rearange_pars(a, b, c, d, a2, b2, c2, d2)
    est_bounds = estimate_bounds(x, y, est_pars, usetol=apply_tol,
                                 tolerance=tolerance)

    return curve_fitting(x, y, func, est_pars, est_bounds)



def analyze_sig_function(x, y, pars, model='sigmoid'):
    """Perform a curve analysis on a sigmoid model.

    A sigmoid model is analyzed using the given parameters.

    Parameters
    ----------
    x : 1D-array
        Values corresponding to the horizontal axis.
    y : 1D-array
        Values corresponding to the vertical axis.
    pars : [float]
        Optimal parameters for the defined model.
    model : str, optional
        Model function to be analyzed. "sigmoid", "boltzmann" and "gompertz"
        are available. The default is "sigmoid".

    Raises
    ------
    ValueError
        Raised when the model is unknown.

    Returns
    -------
    dict
        A dictionary of determined critical points and more.
        slope : float
            Slope at the poi.
        poi_x : float
            Horizontal coordinate of the point of inflection.
        poi_y : float
            Vertical coordinate of the point of inflection.
        peak_y : float
            Vertical coordinate of the peak of the curve.
        intercept : float
            Intersect with the vertical axis.
        start_exp : float
            Start point of the exponential phase defined as the point when the
            vertical coordinate of a tangent at the poi equals the intercept of
            the curve.
        end_exp : float
            End point of the exponential phase defined as the point when the
            vertical coordinate of a tangent at the poi equals the peak of the
            curve.
        len_exp : float
            Length of the exponential phase.
        integral : flaot
            Area below the curve during the exponential phase.
        err : float
            An estimate of the absolute error of the integral.
        integral_div : float
            Integral divided by the length of the exponential phase. Usefull
            as score to measure the overall performance of a growth curve.

    """

    from scipy.integrate import quad

    if model == 'sigmoid':
        def func(x): return sigmoid(x, *pars)
        def func_d(x): return sigmoid_d1(x, *pars[:-1])
        poi_x = pars[1]
        peak_y = pars[2] + pars[3]

    elif model == 'boltzmann':
        def func(x): return boltzmann(x, *pars)
        def func_d(x): return boltzmann_d1(x, *pars)
        poi_x = pars[1]
        peak_y = pars[2]

    elif model == 'gompertz':
        from scipy.optimize import brentq
        def func(x): return gompertz(x, *pars)
        def func_d(x): return gompertz_d1(x, *pars[:-1])
        def func_d2(x): return gompertz_d2(x, *pars[:-1])
        x_root_1, x_root_2 = approximate_root(x, func_d2(x))
        poi_x = brentq(func_d2, x_root_1, x_root_2)
        peak_y = pars[2] + pars[3]

    else: raise ValueError(
        'unknown function. Choose "sigmoid", "boltzmann" or "gompertz"')

    poi_y = func(poi_x)
    slope = func_d(poi_x)
    intercept = pars[3]
    tangent_x, tangent_y = calc_tangent(x, func, func_d, poi_x,
                                        py=(peak_y, intercept))
    start_exp = tangent_x[0]
    end_exp = tangent_x[-1]
    len_exp = abs(end_exp - start_exp)
    integral, err = quad(lambda x: func(x) - tangent_y[0],
                         a=start_exp, b=end_exp)
    integral_div = integral / len_exp

    return {
        'slope':         slope,
        'poi_x':         poi_x,
        'poi_y':         poi_y,
        'peak_y':        peak_y,
        'intercept':     intercept,
        'start_exp':     start_exp,
        'end_exp':       end_exp,
        'len_exp':       len_exp,
        'integral':      integral,
        'err':           err,
        'integral_div':  integral_div,
        }



def analyze_dsig_function(x, y, pars, model='sigmoid'):
    """Perform a curve analysis on a sigmoid model.

    A sigmoid model is analyzed using the given parameters.

    Parameters
    ----------
    x : 1D-array
        Values corresponding to the horizontal axis.
    y : 1D-array
        Values corresponding to the vertical axis.
    pars : [float]
        Optimal parameters for the defined model.
    model : str, optional
        Model function to be analyzed. "sigmoid", "boltzmann" and "gompertz"
        are available. The default is "sigmoid".

    Raises
    ------
    ValueError
        Raised when the model is unknown.

    Returns
    -------
    dict
        A dictionary of determined critical points and more.
        slope : float
            Slope at the poi.
        poi_x : float
            Horizontal coordinate of the point of inflection.
        poi_y : float
            Vertical coordinate of the point of inflection.
        peak_x : float
            Horizontal coordinate of the peak of the curve.
        peak_y : float
            Vertical coordinate of the peak of the curve.
        intercept : float
            Intersect with the vertical axis.
        start_exp : float
            Start point of the exponential phase defined as the point when the
            vertical coordinate of a tangent at the poi equals the intercept of
            the curve.
        end_exp : float
            End point of the exponential phase defined as the point when the
            vertical coordinate of a tangent at the poi equals the peak of the
            curve.
        len_exp : float
            Length of the exponential phase.
        integral : flaot
            Area below the curve during the exponential phase.
        err : float
            An estimate of the absolute error of the integral.
        integral_div : float
            Integral divided by the length of the exponential phase. Usefull
            as score to measure the overall performance of a growth curve.
        slope2 : float
            Slope at the poi after the peak.
        poi_x2 : float
            Horizontal coordinate of the point of inflection after the peak.
        poi_y2 : float
            Vertical coordinate of the point of inflection after the peak.
        plateau : float
            Asymptote of the curve after the peak.
        start_deg : float
            Start point of the degradation phase defined as the point when the
            vertical coordinate of a tangent at the poi after the peak equals
            the peak of the curve.
        end_deg : float
            End point of the degradation phase defined as the point when the
            vertical coordinate of a tangent at the poi after the peak equals
            the asymptote of the curve.
        len_deg : float
            Length of the degradation phase.
        integral2 : flaot
            Area below the curve during the degradation phase.
        err2 : float
            An estimate of the absolute error of the integral.
        integral_div2 : float
            Integral divided by the length of the degradation phase. Usefull
            as score to measure the overall performance of a growth curve.

    """

    from scipy.integrate import quad
    from scipy.optimize import brentq

    if model == 'sigmoid':
        def func(x):
            return sigmoid(x, *pars[:4]) - sigmoid(x, *pars[4:], 0)
        def func_d(x):
            return sigmoid_d1(x, *pars[:3]) - sigmoid_d1(x, *pars[4:])
        def func_d2(x):
            return sigmoid_d2(x, *pars[:3]) - sigmoid_d2(x, *pars[4:])
        def calc_plateau(peak, plat): return peak - plat

    elif model == 'boltzmann':
        a, b, c, d, a2, b2, d2 = pars
        def func(x):
            return boltzmann(x, *pars[:4]) - boltzmann(x, a2, b2, c, d2) + d2
        def func_d(x):
            return boltzmann_d1(x, *pars[:4]) - boltzmann_d1(x, a2, b2, c, d2)
        def func_d2(x):
            return boltzmann_d2(x, *pars[:4]) - boltzmann_d2(x, a2, b2, c, d2)
        def calc_plateau(peak, plat): return plat

    elif model == 'gompertz':
        def func(x):
            return gompertz(x, *pars[:4]) - gompertz(x, *pars[4:], 0)
        def func_d(x):
            return gompertz_d1(x, *pars[:3]) - gompertz_d1(x, *pars[4:])
        def func_d2(x):
            return gompertz_d2(x, *pars[:3]) - gompertz_d2(x, *pars[4:])
        def calc_plateau(peak, plat): return peak - plat

    else: raise ValueError(
        'unknown function. Choose "sigmoid", "boltzmann" or "gompertz"')


    # estimate peak
    est_peak_x = x[y == max(y)]
    # divide x in area before and after the peak
    x_bef_peak, x_aft_peak = x[x <= est_peak_x], x[x > est_peak_x]

    # peak of func_d2 before the peak
    est_peak_d2  = x_bef_peak[func_d2(x_bef_peak) == max(func_d2(x_bef_peak))]
    # peak of func_d2 after the peak
    est_peak2_d2 = x_aft_peak[func_d2(x_aft_peak) == max(func_d2(x_aft_peak))]
    # the maxima of the 2nd derivation are used to limit the range of x to
    # discard potential minima before and after the peak
    x1 = x_bef_peak[x_bef_peak >= est_peak_d2]
    x2 = x_aft_peak[x_aft_peak <= est_peak2_d2]

    poi_x  = brentq(func_d2, *approximate_root(x1, func_d2(x1)))
    poi_x2 = brentq(func_d2, *approximate_root(x2, func_d2(x2)))

    peak_x = brentq(func_d, poi_x, poi_x2)
    peak_y = func(peak_x)
    poi_y  = func(poi_x)
    slope  = func_d(poi_x)
    intercept = pars[3]
    tangent_x,  tangent_y  = calc_tangent(x, func, func_d, poi_x,
                                        py=(peak_y, intercept))
    start_exp = tangent_x[0]
    end_exp = tangent_x[-1]
    len_exp = abs(end_exp - start_exp)
    integral, err = quad(lambda x: func(x) - tangent_y[0],
                         a=start_exp, b=end_exp)
    integral_div = integral / len_exp

    poi_y2 = func(poi_x2)
    slope2 = func_d(poi_x2)
    plateau = calc_plateau(peak_y, pars[6])
    tangent_x2, tangent_y2 = calc_tangent(x, func, func_d, poi_x2,
                                        py=(peak_y, plateau))
    start_deg = tangent_x2[0]
    end_deg = tangent_x2[-1]
    len_deg = abs(end_deg - start_deg)
    integral2, err2 = quad(lambda x: func(x) - tangent_y2[-1],
                           a=start_deg, b=end_deg)
    integral_div2 = integral2 / len_deg


    return {
        'slope':         slope,
        'poi_x':         poi_x,
        'poi_y':         poi_y,
        'peak_x':        peak_x,
        'peak_y':        peak_y,
        'intercept':     intercept,
        'start_exp':     start_exp,
        'end_exp':       end_exp,
        'len_exp':       len_exp,
        'integral':      integral,
        'err':           err,
        'integral_div':  integral_div,
        'slope2':        slope2,
        'poi_x2':        poi_x2,
        'poi_y2':        poi_y2,
        'plateau':       plateau,
        'start_deg':     start_deg,
        'end_deg':       end_deg,
        'len_deg':       len_deg,
        'integral2':     integral2,
        'err2':          err2,
        'integral_div2': integral_div2,
        }