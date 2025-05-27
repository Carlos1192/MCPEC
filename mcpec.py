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

    fit_sig_model(x, y, tolerance=0.1, model='sigmoid'):
        Fit a logistic sigmoid, Boltzmann or Gompertz model to the
        given data.

    fit_dsig_model(x, y, tolerance=0.1, model='sigmoid'):
        Fit a logistic double-sigmoid, double-Boltzmann or
        double-Gompertz model to the given data.

    analyze_sig_model(x, y, pars, model='sigmoid'):
        Perform a curve analysis on a sigmoid model.

    analyze_dsig_model(x, y, pars, model='sigmoid'):
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
    return c / (1+e((b-x)*a)) + d

def boltzmann(x, a, b, c, d):
    # Boltzmann function
    from numpy import exp as e
    return (c-d) / (1+e((b-x)/a)) + d

def gompertz(x, a, b, c, d):
    # Gompertz function
    from numpy import exp as e
    return c*e(-b*e(-a*x)) + d


def dsigmoid(x, a, b, c, d, a2, b2, c2):
    return sigmoid(x, a, b, c, d) - sigmoid(x, a2, b2, c2, 0)

def dboltzmann(x, a, b, c, d, a2, b2, d2):
    return boltzmann(x, a, b, c, d) - boltzmann(x, a2, b2, c, d2) + d2

def dgompertz(x, a, b, c, d, a2, b2, c2):
    return gompertz(x, a, b, c, d) - gompertz(x, a2, b2, c2, 0)



#######################################################################
#######################################################################



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

    dx, dy = x[1:], (y[1:]-y[:-1])/(x[1:]-x[:-1])
    
    peak = max(y)
    pxi = [e for e, i in enumerate(y) if i==max(y)][0]  # peak x-index
    slope1, slope2 = max(dy[:pxi]), min(dy[pxi:])
    poi1 = dx[:pxi][dy[:pxi]==slope1][0]
    poi2 = dx[pxi:][dy[pxi:]==slope2][0]
    intercept = min(abs(y[:pxi]))
    plateau = min(abs(y[pxi:]))
    
    return slope1, slope2, poi1, poi2, peak, intercept, plateau



def curve_fitting(x, y, func, p0, bounds):
    """Fit a model to the given data.

    Adjust the parameters of func to best fit the data using
    scipy.optimize.curve_fit(). The initial guesses p0 are crucial for
    a successful fit. bounds limits the range of the parameters to
    avoid overfitting.

    Parameters
    ----------
    x : 1D-array
        Values corresponding to the horizontal axis.
    y : 1D-array
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
    adj_r2: float
        Adjusted coefficient of determination.
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
    adj_r2 = 1-(1-r2)*((len(y)-1) / (len(y)-len(pars)-1))
    rmsd = ss_res**0.5
    
    return pars, pcov, r2, adj_r2, rmsd



def lack_of_fit_F_test(x, y, pars, model='sigmoid', alpha = 0.05):
    """Perform a lack-of-fit sum of squares test.
    
    
    
    Parameters
    ----------
    x : 1D-array
        Values corresponding to the horizontal axis.
    y : 2D-array
        Replicate values corresponding to the vertical axis.
    pars : [float]
        Parameters of the chosen model.
    model : str, optional
        Model function to be fitted to the data. 'sigmoid', 'boltzmann' and
        'gompertz' are available. The default is 'sigmoid'.
    alpha : float, optional
        Significance level. The default is 0.05.

    Raises
    ------
    ValueError
        Raised when the model is unknown.

    Returns
    -------
    dict
        A dictionary of determined statistical values.
        sse : float
            Total error sum of sqares.
        sslf : float
            Lack-of-fit sum of squares.
        sspe : float
            Pure error sum of squares.
        df_sse : float
            Degrees of freedom of total error.
        df_sslf : float
            Degrees of freedom of lack-of-fit.
        df_sspe : float
            Degrees of freedom of pure error.
        F_stats : float
            Ratio of variability between group means and variability within groups.
        p : float
            Statistical significance.

    """
    if len(pars) == 4:
        if model == 'sigmoid':
            func = sigmoid(x, *pars)
        elif model == 'boltzmann':
            func = boltzmann(x, *pars)
        elif model == 'gompertz':
            func = gompertz(x, *pars)
        else: raise ValueError(
            'unknown function. Choose "sigmoid", "boltzmann" or "gompertz"')
    
    elif len(pars) == 7:
        if model == 'sigmoid':
            func = dsigmoid(x, *pars)
        elif model == 'boltzmann':
            func = dboltzmann(x, *pars)
        elif model == 'gompertz':
            func = dgompertz(x, *pars)
        else: raise ValueError(
            'unknown function. Choose "sigmoid", "boltzmann" or "gompertz"')
    
    else: raise ValueError(
        'number of parameters does not match the defined models')
    
    # error sum of sqares
    # prediction error: distance between measurements and model
    pe = y - func
    sse = sum(pe.flatten()**2)
    # lack of fit sum of sqares
    # total lack of fit: distance between average and model
    lof = sum(y)/len(y) - func
    sslf = sum(lof.flatten()**2)    
    # pure error sum of sqares
    # total error: distance between measurements and average
    te = y - sum(y)/len(y)
    sspe = sum(te.flatten()**2)
    # degrees of freedom
    # total error: number of y values - number of parameters
    df_sse = len(y.flatten()) - len(pars)
    # lack of fit: number of x values - number of parameters
    df_sslf = len(x) - len(pars)
    # prediction error: number of y values - number of x values
    df_sspe = len(y.flatten()) - len(x)
    # mean squares
    mslf, mspe = sslf/df_sslf, sspe/df_sspe
    # F-statistics
    f_st = mslf/mspe
    
    # calculate p-value
    import scipy.stats
    p = 1-scipy.stats.f.cdf(f_st, df_sslf, df_sspe)
    
    if p < alpha:
        print(f'p: {p} < {alpha}: significant lack of fit!!!')
    else:
        print(f'p: {p} > {alpha}: lack of fit not significant')
        
    return {
        'sse':     float(sse),
        'sslf':    float(sslf),
        'sspe':    float(sspe),
        'df_sse':  float(df_sse),
        'df_sslf': float(df_sslf),
        'df_sspe': float(df_sspe),
        'F_stats': float(f_st),
        'p':       float(p),
        }



#######################################################################
#######################################################################



def fit_sig_model(x, y, tolerance=0.1, model='sigmoid'):
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
    adj_r2 : float
        Adjusted coefficient of determination.
    rmsd : float
        Root-mean-square deviation.

    """
    from numpy import inf
    slope, slope2, poi, poi2, peak, intercept, plateau = estimate_pars(x, y)
    x_range = abs(max(x) - min(x)) * tolerance
    y_range = abs(max(y) - min(y)) * tolerance
    
    if model == 'sigmoid':
        func = sigmoid
        p0 = 1, poi, peak, intercept
        bounds = (
            (  0, poi-x_range, peak-y_range, intercept-y_range),
            (inf, poi+x_range, peak+y_range, intercept+y_range)
            )
    
    elif model == 'boltzmann':
        func = boltzmann
        p0 = 1, poi, peak, intercept
        bounds = (
            (  0, poi-x_range, peak-y_range, intercept-y_range),
            (inf, poi+x_range, peak+y_range, intercept+y_range)
            )
    
    elif model == 'gompertz':
        func = gompertz
        p0 = 1, 1, peak, intercept
        bounds = (
            (  0, -inf, peak-y_range, intercept-y_range),
            (inf,  inf, peak+y_range, intercept+y_range)
            )
    
    else: raise ValueError(
        'unknown function. Choose "sigmoid", "boltzmann" or "gompertz"')
    
    return curve_fitting(x, y, func, p0, bounds)



def analyze_sig_model(x, y, pars, model='sigmoid', interval_steps=3):
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
    interval_steps : int, optional
        Defines the size of the window size to find the POI. The default is 3.

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
    
    
    from numpy import exp as e
    from scipy.optimize import brentq
    from scipy.integrate import quad
    
    a, b, c, d = pars
    
    if model == 'sigmoid':
        def sig(x, a, b):
            return 1/(1+e((b-x)*a))
        
        def func(x):
            # reparameterized logistic sigmoid function
            return c * sig(x, a, b) + d
        
        def func_d1(x):
            # simplified first derivative of the logistic sigmoid function
            return c*a * sig(x, a, b)*(1-sig(x, a, b))
        
        def func_d2(x):
            # simplified second derivative of the logistic sigmoid function
            return c*a**2 * sig(x, a, b)*(1-sig(x, a, b)) * (1-2*sig(x, a, b))
        
        
    elif model == 'boltzmann':
        def sig(x, a, b):
            return 1/(1+e((b-x)/a))
        
        def func(x):
            # Boltzmann function
            return (c-d) * sig(x, a, b) + d
        
        def func_d1(x):
            # simplified first derivative of the Boltzmann function
            return (c-d)/a * sig(x, a, b)*(1-sig(x, a, b))
        
        def func_d2(x):
            # simplified second derivative of the Boltzmann function
            return (c-d)/a**2 * sig(x, a, b)*(1-sig(x, a, b)) * (1-2*sig(x, a, b))
        
        
    elif model == 'gompertz':
        def sig(x, a, b):
            return -b*e(-a*x)
        
        def func(x):
            # Gompertz function
            return c*e(sig(x, a, b)) + d
        
        def func_d1(x):
            # simplified first derivative of the Gompertz function
            return func(x, a, b, c, 0) * sig(x, a, b)*(-a)
        
        def func_d2(x):
            # simplified second derivative of the Gompertz function
            return func_d1(x, a, b, c) * (sig(x, a, b)*(-a)-a)
        
        
    else: raise ValueError(
        'unknown function. Choose "sigmoid", "boltzmann" or "gompertz"')
    
    
    # find sign changing indexes (SCIs)
    sci_d2 = [e for e,i in enumerate(func_d2(x)[:-1]*func_d2(x)[1:]) if i<0]
    
    if len(sci_d2) < 1:
        print('No POIs found. Something went somewhere terribly wrong...')
        return {}
    
    
    sample_intervals = float(abs(min(x[:-1] - x[1:])))
    # define windows size to determine roots
    ws = sample_intervals*interval_steps  # window size
    
    # compute POI
    poi_x  = brentq(func_d2, x[sci_d2[0]]-ws, x[sci_d2[0]]+ws)
    poi_y  = func(poi_x)
    
    # compute max slope
    slope  = func_d1(poi_x)
    
    # define intercept as point with a slope < 0.01
    # start counting at POI and go backwards
    intercept_x = poi_x
    while func_d1(intercept_x)*(a/abs(a)) > 0.01:
        intercept_x = intercept_x - sample_intervals
        intercept = func(intercept_x)
    
    # define peak as point with a slope < 0.01
    # start counting at POI and go forward
    peak_x = poi_x
    while func_d1(peak_x)*(a/abs(a)) > 0.01:
        peak_x = peak_x + sample_intervals
        peak_y = func(peak_x)
    
    
    # draw secants through POIs to determine exp. and deg. phase
    # f(x)=ax+b with poi_y=slope*poi_x+b <=> b=poi_y-slope*poi_x
    # intercept = slope*x + (poi_y-slope*poi_x)
    start_exp = (intercept - (poi_y-slope*poi_x)) / slope
    end_exp = (peak_y - (poi_y-slope*poi_x)) / slope
    len_exp = abs(end_exp - start_exp)
    
    # compute definite integrals of exp. and deg. phase
    integral, err = quad(lambda x: func(x)-intercept, a=start_exp, b=end_exp)
    
    integral_div = integral / len_exp
    
    return {
        'slope':         float(slope),
        'poi_x':         float(poi_x),
        'poi_y':         float(poi_y),
        'peak_y':        float(peak_y),
        'intercept':     float(intercept),
        'start_exp':     float(start_exp),
        'end_exp':       float(end_exp),
        'len_exp':       float(len_exp),
        'integral':      float(integral),
        'err':           float(err),
        'integral_div':  float(integral_div),
        }



#######################################################################
#######################################################################



def fit_dsig_model(x, y, tolerance=0.1, model='sigmoid'):
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
    adj_r2: float
        Adjusted coefficient of determination.
    rmsd : float
        Root-mean-square deviation.

    """

    from numpy import inf
    slope1, slope2, poi1, poi2, peak, intercept, plateau = estimate_pars(x, y)
    x_range = abs(max(x) - min(x)) * tolerance
    y_range = abs(max(y) - min(y)) * tolerance
    
    if model == 'sigmoid':
        func = dsigmoid
        p0 = 1, poi1, peak-intercept, intercept, 1, poi2, peak-plateau
        bounds = (
            (  0, poi1-x_range, -inf, intercept-y_range,
               0, poi2-x_range, peak-plateau-y_range),
            (inf, poi1+x_range,  inf, intercept+y_range,
             inf, poi2+x_range, peak-plateau+y_range)
            )
        
    elif model == 'boltzmann':
        func = dboltzmann
        p0 = 1, poi1, peak, intercept, 1, poi2, plateau
        bounds = (
            (  0, poi1-x_range, -inf, intercept-y_range,
               0, poi2-x_range, plateau-y_range),
            (inf, poi1+x_range,  inf, intercept+y_range,
             inf, poi2+x_range, plateau+y_range)
            )
        
    elif model == 'gompertz':
        func = dgompertz
        p0 = 1, poi1, peak, intercept, 1, poi2, plateau
        bounds = (
            (  0, -inf, -inf, intercept-y_range,   0, -inf, plateau-y_range),
            (inf,  inf,  inf, intercept+y_range, inf,  inf, plateau+y_range)
            )
        
    else: raise ValueError(
        'unknown function. Choose "sigmoid", "boltzmann" or "gompertz"')
    
    return curve_fitting(x, y, func, p0, bounds)



def analyze_dsig_model(x, y, pars, model='sigmoid', interval_steps=3):
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
    interval_steps : int, optional
        Defines the window size to find POIs. The default is 3.

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

    from numpy import exp as e
    from numpy import arange
    from scipy.optimize import brentq
    from scipy.integrate import quad
    
    
    if model == 'sigmoid':
        def sig(x, a, b):
            return 1/(1+e((b-x)*a))
        
        def sigmoid(x, a, b, c, d):
            # reparameterized logistic sigmoid function
            return c * sig(x, a, b) + d
        
        def sigmoid_d1(x, a, b, c):
            # simplified first derivative of the logistic sigmoid function
            return c*a * sig(x, a, b)*(1-sig(x, a, b))
        
        def sigmoid_d2(x, a, b, c):
            # simplified second derivative of the logistic sigmoid function
            return c*a**2 * sig(x, a, b)*(1-sig(x, a, b)) * (1-2*sig(x, a, b))
        
        def func(x):
            return sigmoid(x, *pars[:4]) - sigmoid(x, *pars[4:], 0)
        
        def func_d1(x):
            return sigmoid_d1(x, *pars[:3]) - sigmoid_d1(x, *pars[4:])
        
        def func_d2(x):
            return sigmoid_d2(x, *pars[:3]) - sigmoid_d2(x, *pars[4:])
        
        
    elif model == 'boltzmann':
        def sig(x, a, b):
            return 1/(1+e((b-x)/a))
        
        def boltzmann(x, a, b, c, d):
            # Boltzmann function
            return (c-d) * sig(x, a, b) + d
        
        def boltzmann_d1(x, a, b, c, d):
            # simplified first derivative of the Boltzmann function
            return (c-d)/a * sig(x, a, b)*(1-sig(x, a, b))
        
        def boltzmann_d2(x, a, b, c, d):
            # simplified second derivative of the Boltzmann function
            return (c-d)/a**2 * sig(x, a, b)*(1-sig(x, a, b)) * (1-2*sig(x, a, b))
        
        def func(x):
            return boltzmann(x, *pars[:4]) - boltzmann(x, pars[4], pars[5], pars[2], pars[6]) + pars[6]
        
        def func_d1(x):
            return boltzmann_d1(x, *pars[:4]) - boltzmann_d1(x, pars[4], pars[5], pars[2], pars[6])
        
        def func_d2(x):
            return boltzmann_d2(x, *pars[:4]) - boltzmann_d2(x, pars[4], pars[5], pars[2], pars[6])
        
        
    elif model == 'gompertz':
        def sig(x, a, b):
            return -b*e(-a*x)
        
        def gompertz(x, a, b, c, d):
            # Gompertz function
            return c*e(sig(x, a, b)) + d
        
        def gompertz_d1(x, a, b, c):
            # simplified first derivative of the Gompertz function
            return gompertz(x, a, b, c, 0) * sig(x, a, b)*(-a)
        
        def gompertz_d2(x, a, b, c):
            # simplified second derivative of the Gompertz function
            return gompertz_d1(x, a, b, c) * (sig(x, a, b)*(-a)-a)
        
        def func(x):
            return gompertz(x, *pars[:4]) - gompertz(x, *pars[4:], 0)
        
        def func_d1(x):
            return gompertz_d1(x, *pars[:3]) - gompertz_d1(x, *pars[4:])
        
        def func_d2(x):
            return gompertz_d2(x, *pars[:3]) - gompertz_d2(x, *pars[4:])
        
        
    else: raise ValueError(
        'unknown function. Choose "sigmoid", "boltzmann" or "gompertz"')
    
    
    # elongate x axis to find POIs if necessary
    sample_intervals = float(abs(min(x[:-1] - x[1:])))
    # elongate x if 1st POI before x[0]
    start_x = min((pars[1]-x[0])*1.5, x[0])
    # elongate x if 2nd POI after x[-1]
    end_x = max(x[-1]+(pars[5]-x[-1])*1.5, x[-1])
    # new x array
    x = arange(start_x, end_x, sample_intervals)
    
    
    # find sign changing indexes (SCIs)
    sci_d1 = [e for e,i in enumerate(func_d1(x)[:-1]*func_d1(x)[1:]) if i<0]
    sci_d2 = [e for e,i in enumerate(func_d2(x)[:-1]*func_d2(x)[1:]) if i<0]
    
    if len(sci_d2) < 2:
        print('No POIs found. Something went somewhere terribly wrong...')
        return {}
    
    # remove 1st POI if its slope is < 0
    if func_d1(x[sci_d2[0]])*(pars[0]/abs(pars[0])) < 0:
        sci_d2 = sci_d2[1:]
    
    # keep peak index between POIs
    sci_d1 = [i for i in sci_d1 if sci_d2[0] < i < sci_d2[-1]]
    
    # define windows size to determine roots
    ws = sample_intervals*interval_steps  # window size
    # if the slopes have different signs the curves are stacked
    # the peak becomes a POI
    # if pars[0]*pars[4] < 0:
    if len(sci_d1) < 1:
        poi_x  = brentq(func_d2, x[sci_d2[0]]-ws, x[sci_d2[0]]+ws)
        peak_x = brentq(func_d2, x[sci_d2[1]]-ws, x[sci_d2[1]]+ws)
        poi_x2 = brentq(func_d2, x[sci_d2[2]]-ws, x[sci_d2[2]]+ws)
    else:
        peak_x = brentq(func_d1, x[sci_d1[0]]-ws, x[sci_d1[0]]+ws)
        poi_x  = brentq(func_d2, x[sci_d2[0]]-ws, x[sci_d2[0]]+ws)
        poi_x2 = brentq(func_d2, x[sci_d2[1]]-ws, x[sci_d2[1]]+ws)
    
    
    # define intercept as point with a slope < 0.01
    # start counting at 1st POI and go backwards
    intercept_x = poi_x
    while abs(func_d1(intercept_x)) > 0.01:
        intercept_x = intercept_x - sample_intervals
        intercept = func(intercept_x)
    
    # define plateau as point with a slope > -0.01
    # start counting at 2nd POI and go forward
    plateau_x = poi_x2
    while abs(func_d1(plateau_x)) > 0.01:
        plateau_x = plateau_x + sample_intervals
        plateau = func(plateau_x)
    
    
    # calculate characteristic points
    poi_y  = func(poi_x)
    poi_y2 = func(poi_x2)
    peak_y = func(peak_x)
    
    slope  = func_d1(poi_x)
    slope2 = func_d1(poi_x2)
    
    # draw secants through POIs to determine exp. and deg. phase
    # f(x)=ax+b with poi_y=slope*poi_x+b <=> b=poi_y-slope*poi_x
    # intercept = slope*x + (poi_y-slope*poi_x)
    start_exp = (intercept - (poi_y-slope*poi_x)) / slope
    end_exp = (peak_y - (poi_y-slope*poi_x)) / slope
    len_exp = abs(end_exp - start_exp)
    
    start_deg = (peak_y - (poi_y2-slope2*poi_x2)) / slope2
    end_deg = (plateau - (poi_y2-slope2*poi_x2)) / slope2
    len_deg = abs(end_deg - start_deg)
    
    # compute definite integrals of exp. and deg. phase
    integral, err = quad(lambda x: func(x)-min(intercept, peak_y), a=start_exp, b=end_exp)
    integral2, err2 = quad(lambda x: func(x)-min(plateau, peak_y), a=start_deg, b=end_deg)
    
    integral_div = integral / len_exp
    integral_div2 = integral2 / len_deg
    
    return {
        'slope':         float(slope),
        'poi_x':         float(poi_x),
        'poi_y':         float(poi_y),
        'peak_x':        float(peak_x),
        'peak_y':        float(peak_y),
        'intercept':     float(intercept),
        'start_exp':     float(start_exp),
        'end_exp':       float(end_exp),
        'len_exp':       float(len_exp),
        'integral':      float(integral),
        'err':           float(err),
        'integral_div':  float(integral_div),
        'slope2':        float(slope2),
        'poi_x2':        float(poi_x2),
        'poi_y2':        float(poi_y2),
        'plateau':       float(plateau),
        'start_deg':     float(start_deg),
        'end_deg':       float(end_deg),
        'len_deg':       float(len_deg),
        'integral2':     float(integral2),
        'err2':          float(err2),
        'integral_div2': float(integral_div2),
        }