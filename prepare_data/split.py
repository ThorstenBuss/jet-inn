import sys
import os

import pandas as pd

__path__ = 'data/raw/top_samples'
__signal_col__   = 'is_signal_new'
__in_file__ = os.path.join(__path__, 'top.h5')

os.makedirs(__path__, exist_ok=True)

if len(sys.argv)==2:
    __in_file__ = sys.argv[1]

df = pd.read_hdf(
    __in_file__,
    key='table'
)

df_bkg = df[df[__signal_col__]==0]
df_sig = df[df[__signal_col__]==1]

df_bkg.set_index(pd.RangeIndex(0,len(df_bkg)),inplace=True)
df_sig.set_index(pd.RangeIndex(0,len(df_sig)),inplace=True)

df_bkg.to_hdf(
    os.path.join(__path__, 'top-bkg.h5'),
    key='table',
    format='table',
    complib = 'blosc',
    complevel=5)

df_sig.to_hdf(
    os.path.join(__path__, 'top-sig.h5'),
    key='table',
    format='table',
    complib = 'blosc',
    complevel=5)
