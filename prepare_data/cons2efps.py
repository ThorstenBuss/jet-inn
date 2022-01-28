import time
import os
import numpy as np
import pandas as pd
import energyflow as ef

__prefix__       = 'data'
__batch_size__   = 2048
__num_events__   = 140000
__num_constits__ = 200
__signal_col__   = 'is_signal_new'
__mass_col__     = 'mass'
__time_start__   = time.time()

list = [
    (
        'raw/top_samples/top-bkg.h5',
        'efp/top_samples/top-efp-bkg.h5'
    ),
    (
        'raw/top_samples/top-sig.h5',
        'efp/top_samples/top-efp-sig.h5'
    )
]

def print_time (msg):
    print(f'[{time.time()-__time_start__:8.2f}]: {msg}')

for in_file, out_file in list:
    print_time('in:  {}'.format(in_file))
    print_time('out: {}'.format(out_file))

    in_file = os.path.join(__prefix__, in_file)
    out_file = os.path.join(__prefix__, out_file)

    out_dir = '/'.join(out_file.split('/')[0:-1])
    os.makedirs(out_dir, exist_ok=True)

    start_id = 0

    while start_id<__num_events__:

        df = pd.read_hdf(
            in_file,
            key='table',
            start=start_id,
            stop=min(start_id+__batch_size__, __num_events__)
        )

        if df.shape[0] == 0:
            break

        feat_list =  ['E','PX','PY','PZ'] 
        cols = ['{0}_{1}'.format(feature,constit)
                for constit in range(__num_constits__)
                for feature in feat_list]
        constit = df[cols].to_numpy().reshape(-1, __num_constits__, len(feat_list))
        isig = df[__signal_col__].to_numpy()

        # 'd<=3' up to third order
        # 'p<=1' remove EFP's that are a product of EFP's
        # 'd>=1' remove constant EFP
        efpset = ef.EFPSet('d<=3', 'p<=1', 'd>=1', measure='hadr', coords='epxpypz')
        masked = [x[x[:,0] > 0] for x in constit]
        efp_vals = efpset.batch_compute(masked)

        out_cols = (['efp_{0}'.format(i+1) for i in range(efp_vals.shape[1])]
        + [__signal_col__])

        df_out = pd.DataFrame(
            data=np.concatenate((efp_vals,isig[:,None]),axis=1),
            index=np.arange(start_id,start_id+len(isig)),
            columns=out_cols)

        df_out.to_hdf(out_file,
            key='table', append=(start_id!=0),
            format='table',
            complib = 'blosc',
            complevel=5)
        
        print_time('from {} to {}'.format(start_id, start_id+len(isig)))
        start_id += __batch_size__
