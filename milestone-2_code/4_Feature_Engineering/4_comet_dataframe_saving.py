#!/usr/bin/env python
# coding: utf-8


from comet_ml import Experiment
import pandas as pd
import os

df = pd.read_csv('feat_engineering_ii_data.csv',index_col=0)

# Filter to get just the required game
subset_df = df[df['gameId'] == 2017021065]

# Log as comet experiment
experiment = Experiment(
    api_key=os.environ.get('COMET_API_KEY'),
    project_name='feature_engineering_data',
    workspace='stwhitfield',
)
experiment.log_dataframe_profile(
subset_df, 
name='wpg_v_wsh_2017021065',  # keep this name
dataframe_format='csv'  # ensure you set this flag!
)
