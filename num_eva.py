# Numerical evaluation of BioLector data


# import csv file
from mcpec import load_data
x, sam, chan, cat = load_data('example_data.csv',
                              sep=';', decimal=',', sort='c')


# sort data
wells = (
    [ 0,  1,  2,  3],  # King crab
    [ 8,  9, 10, 11],  # Snow crab
    [16, 17, 18, 19],  # Chinese mitten crab
    [20, 21, 22, 23],  # Gray shrimp
    [ 4,  5,  6,  7],  # demin. Chinese mitten crab
    [12, 13, 14, 15],  # demin. Gray shrimp
    [24, 25, 26, 27],  # Pur. chitin
    )
samples = [cat[0][i] for i in wells]  # choose data channel


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
    # (253, 185,  19),  # orange
    (245, 130,  32),  # darker orange
    (124,  21,  77),  # dark red
    (  0,   0,   0),  # black
    )
colors = [[i/255 for i in c] for c in colors]



# plot means
from matplotlib.pyplot import subplots, subplots_adjust, legend, savefig, close

rows, columns = (1, 1)
fig, ax = subplots(rows, columns, sharex=True, sharey=True)
fig.set_figwidth(9.6)
fig.set_figheight(5.4)

# set the spacing between subplots
subplots_adjust(wspace=0.1, hspace=0.2)

ax.set_xlim(0, 48)
ax.set_ylim(-50, 400)
ax.set_xlabel('Time [h]')
ax.set_ylabel('Light scatter (gain 25) [-]')
# ax.set_title('Test 1', fontsize=10)
# plot background grid
ax.grid(visible=True, which='major', axis='both', color=colors[2])

for e,bs in enumerate(blanked_samples):
    # plot means +/- SD
    ax.errorbar(x, bs[-2], yerr=bs[-1], elinewidth=0.5, color=colors[e])
    # plot scatter for labels
    ax.scatter(-10, 0, label=labels[e], color=colors[e])

legend(ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.3))
savefig('plot_means.svg', transparent=True, bbox_inches='tight')
close()


#######################################################################
#######################################################################



# fit models
from mcpec import fit_sig_model, fit_dsig_model
def fit_models(x, y, label):
    best_fit, mod = [-1]*5, 0

    # Try to fit all sigmoid models
    for m in ('sigmoid', 'boltzmann', 'gompertz'):
        # tolerance: range of boundaries for initial estimations
        for tol in (0.05, 0.1, 0.2, 0.5, 1):
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

    # Try to fit all double-sigmoid models
    for m in ('sigmoid', 'boltzmann', 'gompertz'):
        # tolerance: range of boundaries for initial estimations
        for tol in (0.05, 0.1, 0.2, 0.5, 1):
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

    return {
        'label':label, 'model':mod, 'pars':best_fit[0], #'pcov':best_fit[1],
        'R2':float(best_fit[2]), 'adjR2':float(best_fit[3]),
        'RMSD':float(best_fit[4])
        }


# fit models to means of samples
fits = [fit_models(x, y[-2], l) for y,l in zip(blanked_samples, labels)]



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

analysis = [analyze_model(x, y[-2], f['pars'], f['model']) for y,f in zip(blanked_samples, fits)]


# save num_eva to CSV
# add fits results to analysis
for e,a in enumerate(analysis):
    analysis[e]['R2'] = fits[e]['R2']
    analysis[e]['adjR2'] = fits[e]['adjR2']

from pandas import DataFrame
df = DataFrame(analysis, columns=analysis[0].keys(), index=labels).transpose()
df.to_csv('num_eva_results.csv', sep=';', decimal=',')



#######################################################################
#######################################################################



# visualize models and curve analysis
def plot_models(x, pars, model):
    from mcpec import sigmoid, boltzmann, gompertz
    from mcpec import dsigmoid, dboltzmann, dgompertz

    if len(pars) == 4:
        if model == 'sigmoid':
            return sigmoid(x, *pars)
        if model == 'boltzmann':
            return boltzmann(x, *pars)
        if model == 'gompertz':
            return gompertz(x, *pars)

    elif len(pars) == 7:
        if model == 'sigmoid':
            return dsigmoid(x, *pars)
        if model == 'boltzmann':
            return dboltzmann(x, *pars)
        if model == 'gompertz':
            return dgompertz(x, *pars)

    else:
        print('Expected parameters do not match models...')
        return False


# compute curves of fitted models
models = [plot_models(x, f['pars'], f['model']) for f in fits]



# adjust figure parameters
rows, columns = (len(blanked_samples), 2)
fig, ax = subplots(rows, columns, sharex=True, sharey=True)
fig.set_figwidth(9.6*columns)
fig.set_figheight(5.4*rows)

# set the spacing between subplots
subplots_adjust(wspace=0.1, hspace=0.2)

ax[0, 0].set_xlim(0, 48)
ax[0, 0].set_ylim(-50, 400)
ax[rows-1, 0].set_xlabel('Time [h]')
ax[rows-1, 1].set_xlabel('Time [h]')

for e,bs in enumerate(blanked_samples):
    ax[e, 0].set_title(labels[e], fontsize=10)
    ax[e, 1].set_title(f'Fitted {fits[e]["model"]} model', fontsize=10)
    ax[e, 0].set_ylabel('Light scatter (gain 25) [-]')

    # plot background grid
    ax[e, 0].grid(visible=True, which='major', axis='both', color=colors[2])
    ax[e, 1].grid(visible=True, which='major', axis='both', color=colors[2])

    # plot replicates
    ax[e, 0].plot(x, bs[0], color=colors[0])
    ax[e, 0].plot(x, bs[1], color=colors[1])
    ax[e, 0].plot(x, bs[2], color=colors[2])

    # plot models
    ax[e, 0].plot(x, models[e], color='black')
    ax[e, 1].plot(x, models[e], color='black')

    # plot scatter for labels
    ax[e, 0].scatter(-10, 0, label='replicate 1', color=colors[0])
    ax[e, 0].scatter(-10, 0, label='replicate 2', color=colors[1])
    ax[e, 0].scatter(-10, 0, label='replicate 3', color=colors[2])
    ax[e, 0].scatter(-10, 0, label='model', color='black')

    # plot analysis results
    ax[e, 1].scatter(analysis[e]['poi_x'], analysis[e]['poi_y'], color=colors[1], label='POIs')
    ax[e, 1].scatter(2, analysis[e]['intercept'], color=colors[7], label='Intercept/Plateau')

    # highlight exp phase
    fill_x = [enu for enu,i in enumerate(x) if analysis[e]['start_exp']<=i<=analysis[e]['end_exp']]
    ax[e, 1].fill_between(x[fill_x], y1=models[e][fill_x], y2=[analysis[e]['intercept']]*len(fill_x), color=colors[2])

    if len(fits[e]['pars']) == 4:
        ax[e, 1].scatter(40, analysis[e]['peak_y'], color=colors[7])

    if len(fits[e]['pars']) == 7:
        ax[e, 1].scatter(analysis[e]['peak_x'], analysis[e]['peak_y'], color=colors[3], label='Peak')
        ax[e, 1].scatter(analysis[e]['poi_x2'], analysis[e]['poi_y2'], color=colors[1])
        ax[e, 1].scatter(40, analysis[e]['plateau'], color=colors[7])
        ax[e, 1].plot([analysis[e]['start_deg'], analysis[e]['end_deg']], [analysis[e]['peak_y'], analysis[e]['plateau']], color='black')

        # highlight deg phase
        fill_x = [enu for enu,i in enumerate(x) if analysis[e]['start_deg']<=i<=analysis[e]['end_deg']]
        ax[e, 1].fill_between(x[fill_x], y1=models[e][fill_x], y2=[min(analysis[e]['plateau'], analysis[e]['peak_y'])]*len(fill_x), color=colors[2])

    ax[e, 1].plot([analysis[e]['start_exp'], analysis[e]['end_exp']], [analysis[e]['intercept'], analysis[e]['peak_y']], color='black', label='Slopes')


ax[0, 0].legend(ncol=1, frameon=True, loc='upper left')
ax[0, 1].legend(ncol=1, frameon=True, loc='upper left')


savefig('num_eva_visual_analysis.pdf', transparent=True, bbox_inches='tight')
close()

