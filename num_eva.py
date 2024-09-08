"""Numerical evaluation of multichannel data using mcpec"""


# import modules
import matplotlib.pyplot as plt



# import csv file
from mcpec import load_data
x, sam, chan, cat = load_data('raw_data.csv',
                              sep=';', decimal=',', sort='c')



# sort data
# sort wells as replicates + blank
wells = (
    [ 0,  1,  2,  3],  # King crab
    [ 8,  9, 10, 11],  # Snow crab
    [16, 17, 18, 19],  # Chinese mitten crab
    [20, 21, 22, 23],  # Gray shrimp
    [ 4,  5,  6,  7],  # demin. Chinese mitten crab
    [12, 13, 14, 15],  # demin. Gray shrimp
    [24, 25, 26, 27],  # Pur. chitin
    [28, 29, 30, 31],  # control
    )

samples = [cat[0][i] for i in wells]



# subtract blanks and calculate mean and SD
def blanking(var, subtract_blank=True):
    from numpy import array, mean, std
    # shift curves to 0
    var = var - [[i[0]] for i in var]
    if subtract_blank:
        blank = var[:-1] - var[-1:]
    else:
        blank = var[:-1]  # proceed without blanking
    blank = array([mean(blank, axis=0), std(blank, axis=0)])
    return blank

blanked_samples = [blanking(i) for i in samples]



# sample labels
labels = (
    '(1) King crab',
    '(2) Snow crab',
    '(3) Chinese mitten crab',
    '(4) Gray shrimp',
    '(D3) Demin. Chinese mitten crab',
    '(D4) Demin. Gray shrimp',
    '(R) Pur. chitin',
    '(C) control',
    )



# colors [R, G, B, A] (Fraunhofer colors)
colors = (
    ( 23, 156, 125),  # green
    (  0,  91, 127),  # blue
    (166, 187, 200),  # gray
    (187,   0,  86),  # red
    (  0, 133, 152),  # dark blue
    ( 57, 193, 205),  # turquoise
    (245, 130,  32),  # orange
    (178, 210,  53),  # light green
    (  0,   0,   0),  # black
    )

colors = [[i/255 for i in c] for c in colors]



# plot the data
plt.xlabel('Time [h]')
plt.ylabel('Light scatter [u]')
plt.xlim(0, 48)
plt.ylim(-50, 400)
for bs, l, c in zip(blanked_samples, labels, colors):
    plt.errorbar(x, bs[0], yerr=bs[1], elinewidth=0.5, color=c)
    plt.scatter(-10, 0, label=l, color=c)
plt.legend(ncol=1, loc='upper right', frameon=False, fontsize=7)
plt.savefig('num_eva_plot.svg', transparent=True, bbox_inches='tight')
# plt.savefig('num_eva_plot.pdf', transparent=True, bbox_inches='tight')



#######################################################################
#######################################################################

# fit models

def fit_model(x, y, tol=0.1, model='sigmoid', func='s'):
    from mcpec import fit_sig_function, fit_dsig_function
    if func == 's':  # fit simple sigmoid models
        return fit_sig_function(x, y, tolerance=tol, model=model)
    else:  # fit double-sigmoid models
        return fit_dsig_function(x, y, tolerance=tol, model=model)


def analyze_model(x, y, pars, model='sigmoid'):
    from mcpec import analyze_sig_function, analyze_dsig_function
    if len(pars) == 4:  # analyze simple sigmoid models
        return analyze_sig_function(x, y, pars, model=model)
    elif len(pars) == 7:  # analyze double-sigmoid models
        return analyze_dsig_function(x, y, pars, model=model)
    else:
        raise ValueError('"pars" contains too many or too few elements')


def fit_and_analyze(x, y, label):
    best_fit = [-1]*5
    for function in ('s', 'd'):
        for model in ('sigmoid', 'boltzmann', 'gompertz'):
            for tolerance in (0.1, 0.2, 0.5):
                try:
                    fit = fit_model(x, y, tol=tolerance,
                                    model=model, func=function)
                    # RMSD decrease by 5% and adj_R2 bigger
                    if fit[4] / best_fit[4] < 0.95 and best_fit[3] < fit[3]:
                        best_fit = fit
                        mod = model
                        print(f'{label[:4]}: func:{mod[:3]}, '
                              f'RMSW:{best_fit[4]:.3f}, R2:{best_fit[2]:.3f}, '
                              f'adj_R2:{best_fit[3]:.3f}')
                except:
                    continue

    analysis = analyze_model(x, y, best_fit[0], model=mod)

    return {
        'label':label, 'model':mod, 'pars':best_fit[0], #'pcov':best_fit[1],
        'R2':best_fit[2], 'adjR2':best_fit[3], 'RMSD':best_fit[4], **analysis
        }



sb = [s[:-1] - s[-1] for s in samples[:-1]]  # subtract blanks
labeled_samples = [(l, s) for l, sub in zip(labels, sb) for s in sub]
results = [fit_and_analyze(x, ls[1], ls[0]) for ls in labeled_samples]

from pandas import DataFrame
DataFrame(results).to_csv('num_eva_results.csv', sep=';', decimal=',')

