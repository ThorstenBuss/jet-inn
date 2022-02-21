import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.default'] = 'rm'
plt.rcParams['text.usetex'] = True

import util

def get_fonts (label_size=14, axislabel_size=22, tick_size=16) :
    labelfont = FontProperties()
    labelfont.set_family('serif')
    labelfont.set_name('Times New Roman')
    labelfont.set_size(label_size)

    axislabelfont = FontProperties()
    axislabelfont.set_family('serif')
    axislabelfont.set_name('Times New Roman')
    axislabelfont.set_size(axislabel_size)

    tickfont = FontProperties()
    tickfont.set_family('serif')
    tickfont.set_name('Times New Roman')
    tickfont.set_size(tick_size)

    return labelfont, axislabelfont, tickfont

def plot_rocs (zs_list, lbs_list, name_list, file,
        text=None,
        text2=None,
        text_pos=(0.05,1.5),
        text2_pos=(0.05,3.5),
        flip=False,
        max_pow=3,
        num_dashed=0):

    fig, axs = plt.subplots( 1, 1, figsize=(4.8,4.3) )
    labelfont, axislabelfont, tickfont = get_fonts()

    tpr_z = np.linspace(0,1,1000)
    axs.plot( tpr_z, 1/(tpr_z+1e-12), 'k--', alpha=0.5 )

    for i, (zs, lbs, name) in enumerate(zip(zs_list, lbs_list, name_list)):
        fpr_z, tpr_z, auc_z = util.calc_roc(lbs, zs, flip=flip)

        if i < len(zs_list)-num_dashed:
            axs.plot( tpr_z, 1/(fpr_z+1e-12),
                alpha=1.0,
                linewidth=1.5,
                label=f'{name:s}, AUC: {auc_z:.2f}')
        else:
            axs.plot( tpr_z, 1/(fpr_z+1e-12),
                alpha=1.0,
                linewidth=1.5,
                color='black',
                label=f'{name}, AUC: {auc_z:.2f}',
                linestyle= 'dashdot' )

    axs.grid( which='both', alpha=0.5 )
    axs.set_xlim(0,1)
        
    axs.set_xlabel('$\epsilon_s$', fontproperties=axislabelfont )
    axs.set_ylabel('$\epsilon_b^{-1}$', fontproperties=axislabelfont )

    axs.set_yscale('log')

    axs.set_xticks([np.round(i*0.2,1) for i in range(6)])
    axs.set_xticklabels([np.round(i*0.2,1) for i in range(6)], fontproperties=tickfont )

    axs.set_ylim((1,1000))

    axs.set_yticks( [10**(i+1) for i in range(max_pow)] )
    axs.set_yticklabels( [ f'$10^{i+1:d}$' for i in range(max_pow) ] , fontproperties=tickfont, va='top' )

    axs.legend( loc='upper right', prop=labelfont )

    if text is not None:
        axs.text( text_pos[0], text_pos[1],  text, va='bottom',
            ha='left', fontproperties=tickfont, bbox=dict(facecolor='white', alpha=0.8) )
    
    if text2 is not None:
        axs.text( text2_pos[0], text2_pos[1],  text2, va='bottom',
            ha='left', fontproperties=tickfont, bbox=dict(facecolor='white', alpha=0.8) )

    fig.tight_layout()
    fig.savefig( file, bbox_inches='tight' )

    plt.close(fig)

def plot_calc_remap (data_dir, plot_dir, lable, signal_name):
    os.makedirs( plot_dir, exist_ok=True)

    text_pos=(0.6,33)
    text2_pos=(0.6,15)

    lbs = np.load( os.path.join(data_dir, 'lable.npy') )
    logps = np.load( os.path.join(data_dir, 'losses.npy') )
    phys = np.load( os.path.join(data_dir, 'raw.npy') )
    latent = np.load( os.path.join(data_dir, 'latent.npy') )
    distance = np.linalg.norm(latent, axis=1)

    lbs_list = [lbs for i in range(4)]
    ps_list = [logps + np.sum((ex-1)*np.log(phys+1e-10),axis=1)/8 for ex in [1.0, 1/3, 0]] + [distance]
    remaps = [r'\(x\)', r'\(\sqrt[3]{x}\)', 'log', 'latent']

    plot_rocs (
        ps_list,
        lbs_list,
        name_list=remaps,
        file=os.path.join(plot_dir, f'{signal_name:s}_roc.pdf'),
        text=lable,
        flip=False,
        text_pos=text_pos,
        num_dashed=1
    )

def plot_latent(data_dir, out_file, idx=0):
    labelfont, axislabelfont, tickfont = get_fonts(18,20,20)
    min_v = -4.5
    max_v = 3
    z = np.load(os.path.join(data_dir, 'latent.npy'))
    lable = np.load(os.path.join(data_dir, 'lable.npy'))
    bins = np.linspace(min_v, max_v, 26)
    colors = ['black', 'red']

    fig,axs = plt.subplots(1,1,figsize=(6,4))

    axs.hist(z[lable==0][:,idx], bins, label='background', color=colors[0], histtype='step', density=True)
    axs.hist(z[lable==1][:,idx], bins, label='signal', color=colors[1], histtype='step', density=True)

    axs.set_xlim(min_v, max_v)

    z_ = np.linspace(min_v, max_v, 501)
    p = 1/np.sqrt(2*np.pi)*np.exp(-z_**2/2)
    axs.plot(z_, p, label='prior', color=colors[0])

    axs.legend(prop=labelfont, loc='upper left', frameon=False)
    plt.xticks(fontproperties=tickfont)
    plt.yticks(fontproperties=tickfont)

    axs.set_xlabel(f'\\(z_{{{idx+1}}}\\)', fontproperties=axislabelfont)
    axs.set_ylabel('normalized distribution', fontproperties=axislabelfont)

    fig.tight_layout()
    fig.savefig(out_file, bbox_inches='tight')
    plt.close()

def plot_losses(data_dir, out_file, min_epoch=0, max_epoch=None, text=None):
    labelfont, axislabelfont, tickfont = get_fonts(18,20,20)
    losses_train = np.load(os.path.join(data_dir, 'losses_over_epochs_train.npy'))
    losses_train[losses_train>5.0] = np.inf
    losses_test = np.load(os.path.join(data_dir, 'losses_over_epochs_test.npy'))
    losses_test[losses_test>5.0] = np.inf
    if not max_epoch:
        max_epoch = len(losses_train)
    epochs = np.arange(min_epoch,max_epoch)

    plt.plot(epochs, losses_test[epochs], label='test loss')
    plt.plot(epochs, losses_train[epochs], label='train loss')

    max_v = max(np.max(losses_train), np.max(losses_test))
    min_v = min(np.min(losses_train), np.min(losses_test))

    text_pos = (
        min_v+0.8*(max_v-min_v),
        min_epoch+0.75*(max_epoch-min_epoch))

    if text:
        plt.text( text_pos[1], text_pos[0],  text, va='top',
            ha='left', fontproperties=tickfont, bbox=dict(facecolor='white', alpha=0.8) )

    plt.legend(prop=labelfont)
    plt.xticks(fontproperties=tickfont)
    plt.yticks(fontproperties=tickfont)

    plt.xlabel('epoch', fontproperties=axislabelfont)
    plt.ylabel('loss', fontproperties=axislabelfont)

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def plot_auc(exps, auc_list, label_list):
    labelfont, axislabelfont, tickfont = get_fonts(20,20,20)
    fig,axs = plt.subplots(1,1,figsize=(6,4))
    for auc, label in zip(auc_list, label_list):
        axs.plot(exps, auc, label=label)
    axs.legend(prop=labelfont, frameon=False)
    plt.xticks(fontproperties=tickfont)
    plt.yticks(fontproperties=tickfont)
    axs.set_xlabel(r'\(n\)', fontproperties=axislabelfont)
    axs.set_ylabel('AUC', fontproperties=axislabelfont)
    axs.set_xlim(exps[0], exps[-1])
    axs.set_ylim(0.9, 0.95)
    fig.savefig('plots/auc.pdf', bbox_inches='tight')

def plot_imt(exps, imt_list, label_list):
    labelfont, axislabelfont, tickfont = get_fonts(20,20,20)
    fig,axs = plt.subplots(1,1,figsize=(6,4))
    for imt, label in zip(imt_list, label_list):
        axs.plot(exps, imt, label=label)
    axs.legend(prop=labelfont, frameon=False)
    plt.xticks(fontproperties=tickfont)
    plt.yticks(fontproperties=tickfont)
    axs.set_xlabel(r'\(n\)', fontproperties=axislabelfont)
    axs.set_ylabel(r'\(\epsilon_{b}^{-1}(\epsilon_{s}=0.5)\)', fontproperties=axislabelfont)
    axs.set_xlim(exps[0], exps[-1])
    axs.set_yscale('log')
    axs.set_ylim(1, 1e3)
    fig.savefig('plots/imt.pdf', bbox_inches='tight')

def calc_auc_imt(data_dir):
    exps = np.linspace(0.,1.,1001)
    lbs = np.load( os.path.join(data_dir, "lable.npy") )
    logps = np.load( os.path.join(data_dir, "losses.npy") )
    phys = np.load( os.path.join(data_dir, "raw.npy") )

    ps_list = [logps + np.sum((ex-1)*np.log(phys+1e-10),axis=1)/8 for ex in exps]
    perf_stats_list = [util.get_perf_stats(lbs, measures, flip=False,pos=0.5) for measures in ps_list]

    auc = np.array([perf_stats[0] for perf_stats in perf_stats_list])
    imt = np.array([perf_stats[1] for perf_stats in perf_stats_list])

    return exps, auc, imt

def main():
    exps, auc_top, imt_top = calc_auc_imt('results/top')
    exps, auc_qcd, imt_qcd = calc_auc_imt('results/qcd')
    plot_auc(exps, [auc_top, auc_qcd], ['Top', 'QCD'])
    plot_imt(exps, [imt_top, imt_qcd], ['Top', 'QCD'])

if __name__=='__main__':
    main()
