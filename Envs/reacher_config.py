repeated = {
    'batch_ac_shared_gc': {"gamma_coef":18.94, "scale":143, "lr":0.0042,"hid":8,"buffer":25,
                       "lr_weight":0.0005,"critic_hid":64,"gamma":0.99},
    'weighted_batch_ac': {"gamma_coef":8.9, "scale":133, "lr":0.0047,"hid":32,"buffer":45,
                       "lr_weight":0.0003,"critic_hid":32,"gamma":0.8},
    'biased': {"gamma_coef":11.88, "scale":32, "lr":0.0042,"hid":32,"buffer":25,
                       "lr_weight":0.0023,"critic_hid":64,"gamma":0.8},
    'naive': {"gamma_coef":7.39, "scale":126, "lr":0.0036,"hid":64,"buffer":25,
                       "lr_weight":0.001,"critic_hid":16,"gamma":0.95}
}

episodic = {
    'batch_ac_shared_gc': {"gamma_coef":11.65, "scale":6, "lr":0.0041,"hid":8,"buffer":25,
                       "lr_weight":0.0036,"critic_hid":64,"gamma":0.95},
    'weighted_batch_ac': {"gamma_coef":19.08, "scale":9, "lr":0.003,"hid":32,"buffer":5,
                       "lr_weight":0.0047,"critic_hid":16,"gamma":0.95},
    'biased': {"gamma_coef":16.14, "scale":76, "lr":0.0038,"hid":32,"buffer":5,
                       "lr_weight":0.003,"critic_hid":32,"gamma":0.95},
    'naive': {"gamma_coef":18.46, "scale":29, "lr":0.0046,"hid":32,"buffer":5,
                       "lr_weight":0.0004,"critic_hid":32,"gamma":0.99}
}