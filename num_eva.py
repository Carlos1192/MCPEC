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
    )

samples = [cat[0][i] for i in wells]



# subtract blanks and calculate mean and SD
def blanking(var, subtract_blank=True):
    from numpy import array, mean, std
    if subtract_blank:
        # shift curves to 0
        var = var - [[mean(i[:10])] for i in var]
        blank = var[:-1] - var[-1:]
    else:
        blank = var[:-1]  # proceed without blanking
    return array([*blank, mean(blank, axis=0), std(blank, axis=0)])

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



# plot the data
plt.xlabel('Time [h]')
plt.ylabel('Light scatter [u]')
plt.xlim(0, 48)
plt.ylim(-50, 400)
for bs, l, c in zip(blanked_samples, labels, colors):
    plt.errorbar(x, bs[-2], yerr=bs[-1], elinewidth=0.5, color=c)
    plt.scatter(-10, 0, label=l, color=c)
plt.legend(ncol=1, loc='upper right', frameon=False, fontsize=7)
plt.savefig('num_eva_plot.svg', transparent=True, bbox_inches='tight')
# plt.savefig('num_eva_plot.pdf', transparent=True, bbox_inches='tight')



#######################################################################
#######################################################################



# fit sigmoid models
from mcpec import fit_sig_model, fit_dsig_model
def fit_models(x, y, label):
    
    best_fit, mod = [-1]*5, 0
    # fit single and double-sigmoid models
    for model_type in ('s', 'd'):
        # try all sigmoid models
        for m in ('sigmoid', 'boltzmann', 'gompertz'):
            # tolerance: range of boundaries for initial estimations
            for tol in (0.05, 0.1, 0.2, 0.5, 1):
                
                # fit single-sigmoid models
                if model_type == 's':
                    try:
                        fit = fit_sig_model(x, y, tolerance=tol, model=m)
                        # RMSD decrease by 5% and adj_R2 bigger
                        if fit[4] / best_fit[4] < 0.95 and best_fit[3] < fit[3]:
                            best_fit, mod = fit, m
                            print(f'{label[:4]}: func:{mod[:3]}, '
                                  f'RMSW:{best_fit[4]:.3f}, R2:{best_fit[2]:.3f}, '
                                  f'adj_R2:{best_fit[3]:.3f}')
                    except:
                        continue
                
                # fit double-sigmoid models
                elif model_type == 'd':
                    try:
                        fit = fit_dsig_model(x, y, tolerance=tol, model=m)
                        # RMSD decrease by 5% and adj_R2 bigger
                        if fit[4] / best_fit[4] < 0.95 and best_fit[3] < fit[3]:
                            best_fit, mod = fit, m
                            print(f'{label[:4]}: func:{mod[:3]}, '
                                  f'RMSW:{best_fit[4]:.3f}, R2:{best_fit[2]:.3f}, '
                                  f'adj_R2:{best_fit[3]:.3f}')
                    except:
                        continue
                    
                else:
                    raise ValueError('Model not found...')
    
    return {
        'label':label, 'model':mod, 'pars':best_fit[0], #'pcov':best_fit[1],
        'R2':float(best_fit[2]), 'adjR2':float(best_fit[3]),
        'RMSD':float(best_fit[4])
        }



# fit sigmoid models to means of samples
fits = [fit_models(x, y[-2], l) for y, l in zip(blanked_samples, labels)]



#######################################################################
#######################################################################



# analyze fitted sigmoid models
from mcpec import analyze_sig_model, analyze_dsig_model
def analyze_model(x, y, pars, model='sigmoid'):
    if len(pars) == 4:
        return analyze_sig_model(x, y, pars, model=model)
    elif len(pars) == 7:
        return analyze_dsig_model(x, y, pars, model=model)
    else:
        raise ValueError('Model not found...')
    
    return True

analysis = [analyze_model(x, y[-2], f['pars'], f['model']) for y, f in zip(blanked_samples, fits)]


# save data to CSV
from pandas import DataFrame
df = DataFrame(analysis, columns=analysis[0].keys(), index=labels).transpose()
df.to_csv('num_eva_results.csv', sep=',', decimal='.')
