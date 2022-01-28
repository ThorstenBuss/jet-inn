import os

import numpy as np
import torch

from inn import INN
from trainer import Trainer
import plot
import util

def main():
    extargs = util.loud_config()

    use_cuda = not extargs['disable_cuda'] and torch.cuda.is_available()
    print('use_cuda: ', use_cuda)

    os.makedirs(extargs['plot_dir'], exist_ok=True)
    os.makedirs(extargs['result_dir'], exist_ok=True)

    trainer = Trainer(
            data_bkg = os.path.join(extargs['prefix'], extargs['data_bkg']),
            data_sig = os.path.join(extargs['prefix'], extargs['data_sig']),
            num_samples = extargs['num_samples'],
            num_test = extargs['num_test'],
            batchsize=extargs['batchsize'],
            SB=extargs['SB'],
            use_cuda=use_cuda,
            preprocessing=extargs['preprocessing'],
            std=extargs['std']
        )

    model = INN(trainer.num_dim, extargs['num_layers'], extargs['hidden'], extargs['dropout'])
    if use_cuda:
        model = model.cuda()

    trainer.train(
        model,
        disable_tqdm=extargs['disable_progress'],
        num_epochs=extargs['num_epochs'],
        lr=extargs['learning_rate'],
        betas=tuple(extargs['betas']),
        gamma=extargs['gamma'],
        result_dir=extargs['result_dir'],
        use_ste_decay=extargs['use_ste_decay'],
        gamma2=extargs['gamma2'],
        lambda_l2=extargs['lambda']
    )

    test(extargs, model, trainer.test_dataset, use_cuda)

def test(extargs, model, test_dataset, use_cuda):
    model.eval()
    with torch.no_grad():
        x, lable = test_dataset[:]
        if use_cuda:
            x = x.cuda()
            lable = lable.cuda()
        z, log_jac_det = model(x)
        losses = 0.5*torch.sum(z**2, 1) - log_jac_det
        losses /= x.shape[1]

    losses_np = losses.cpu().numpy()
    lable_np = lable.cpu().numpy()
    z_np = z.cpu().numpy()
    log_jac_det_np = log_jac_det.cpu().numpy()

    np.save(os.path.join(extargs['result_dir'], 'losses.npy'), losses_np)
    np.save(os.path.join(extargs['result_dir'], 'lable.npy'), lable_np)
    np.save(os.path.join(extargs['result_dir'], 'latent.npy'), z_np)
    np.save(os.path.join(extargs['result_dir'], 'log_jac_det.npy'), log_jac_det_np)
    np.save(os.path.join(extargs['result_dir'], 'raw.npy'), test_dataset.raw)
    torch.save(model.state_dict(), os.path.join(extargs['result_dir'], 'checkpoint.ckpt'))

    plot.plot_calc_remap (
        extargs['result_dir'],
        extargs['plot_dir'],
        extargs['plot_text'],
        extargs['plot_name']
    )

    plot.plot_losses (
        extargs['result_dir'],
        os.path.join(extargs['plot_dir'], f'{extargs["plot_name"]:s}_loss.pdf'),
        min_epoch=0,
        max_epoch=extargs['num_epochs']
    )

    os.makedirs(os.path.join(extargs['plot_dir'], f'{extargs["plot_name"]:s}_latent'), exist_ok=True)
    for i in range(8):
        plot.plot_latent(
            extargs['result_dir'],
            os.path.join(extargs['plot_dir'], f'{extargs["plot_name"]:s}_latent/latent_{i+1:02d}.pdf'),
            i
        )

if __name__=='__main__':
    main()
