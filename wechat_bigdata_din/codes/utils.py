# -*- coding:utf-8 -*-
"""
@Author: LMC_ZC
"""


def get_user_mask_df(df, colname):
    
    u_group = df[['userid', colname]].groupby('userid')
    user_mask_list = [u_df[0] for u_df in u_group if u_df[1][colname].mean() == 0.0 or u_df[1][colname].mean() == 1.0]
    
    mask = ~df['userid'].isin(user_mask_list)
    df = df[mask]
    return df
