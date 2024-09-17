# Gaussian deblur
python posterior_sample.py +data=ffhq +task=gaussian_deblur_circ +model=edm_unet_adm_dps_ffhq +sampler=pnp_edm \
       sampler.mode=vp_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.3 gpu=0 add_exp_name=anneal-0.9
python posterior_sample.py +data=ffhq +task=gaussian_deblur_circ +model=edm_unet_adm_dps_ffhq +sampler=pnp_edm \
       sampler.mode=ve_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.3 gpu=0 add_exp_name=anneal-0.9
python posterior_sample.py +data=ffhq +task=gaussian_deblur_circ +model=edm_unet_adm_dps_ffhq +sampler=pnp_edm \
       sampler.mode=iddpm_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.3 gpu=0 add_exp_name=anneal-0.9
python posterior_sample.py +data=ffhq +task=gaussian_deblur_circ +model=edm_unet_adm_dps_ffhq +sampler=pnp_edm \
       sampler.mode=edm_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.3 gpu=0 add_exp_name=anneal-0.9

# Motion deblur
python posterior_sample.py +data=ffhq +task=motion_deblur_circ +model=edm_unet_adm_dps_ffhq +sampler=pnp_edm \
       sampler.mode=vp_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.3 gpu=0 add_exp_name=anneal-0.9
python posterior_sample.py +data=ffhq +task=motion_deblur_circ +model=edm_unet_adm_dps_ffhq +sampler=pnp_edm \
       sampler.mode=ve_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.3 gpu=0 add_exp_name=anneal-0.9
python posterior_sample.py +data=ffhq +task=motion_deblur_circ +model=edm_unet_adm_dps_ffhq +sampler=pnp_edm \
       sampler.mode=iddpm_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.3 gpu=0 add_exp_name=anneal-0.9
python posterior_sample.py +data=ffhq +task=motion_deblur_circ +model=edm_unet_adm_dps_ffhq +sampler=pnp_edm \
       sampler.mode=edm_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.3 gpu=0 add_exp_name=anneal-0.9

# Super-resolution
python posterior_sample.py +data=ffhq +task=super_resolution_svd +model=edm_unet_adm_dps_ffhq +sampler=pnp_edm \
       sampler.mode=vp_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.3 gpu=0 add_exp_name=anneal-0.9
python posterior_sample.py +data=ffhq +task=super_resolution_svd +model=edm_unet_adm_dps_ffhq +sampler=pnp_edm \
       sampler.mode=ve_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.3 gpu=0 add_exp_name=anneal-0.9
python posterior_sample.py +data=ffhq +task=super_resolution_svd +model=edm_unet_adm_dps_ffhq +sampler=pnp_edm \
       sampler.mode=iddpm_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.3 gpu=0 add_exp_name=anneal-0.9
python posterior_sample.py +data=ffhq +task=super_resolution_svd +model=edm_unet_adm_dps_ffhq +sampler=pnp_edm \
       sampler.mode=edm_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.3 gpu=0 add_exp_name=anneal-0.9

# Coded diffraction patterns
python posterior_sample.py +data=ffhq_grayscale +task=coded_diffraction_patterns_grayscale +model=edm_unet_adm_gray_ffhq +sampler=pnp_edm \
       sampler.mode=vp sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.1 gpu=0 add_exp_name=anneal-0.9
python posterior_sample.py +data=ffhq_grayscale +task=coded_diffraction_patterns_grayscale +model=edm_unet_adm_gray_ffhq +sampler=pnp_edm \
       sampler.mode=ve sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.1 gpu=0 add_exp_name=anneal-0.9
python posterior_sample.py +data=ffhq_grayscale +task=coded_diffraction_patterns_grayscale +model=edm_unet_adm_gray_ffhq +sampler=pnp_edm \
       sampler.mode=iddpm sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.1 gpu=0 add_exp_name=anneal-0.9
python posterior_sample.py +data=ffhq_grayscale +task=coded_diffraction_patterns_grayscale +model=edm_unet_adm_gray_ffhq +sampler=pnp_edm \
       sampler.mode=edm sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.1 gpu=0 add_exp_name=anneal-0.9

# Phase retrieval
python posterior_sample.py +data=ffhq_grayscale +task=phase_retrieval +model=edm_unet_adm_gray_ffhq +sampler=pnp_edm \
       sampler.mode=vp_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.1 gpu=0 add_exp_name=anneal-0.9 num_runs=4
python posterior_sample.py +data=ffhq_grayscale +task=phase_retrieval +model=edm_unet_adm_gray_ffhq +sampler=pnp_edm \
       sampler.mode=ve_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.1 gpu=0 add_exp_name=anneal-0.9 num_runs=4
python posterior_sample.py +data=ffhq_grayscale +task=phase_retrieval +model=edm_unet_adm_gray_ffhq +sampler=pnp_edm \
       sampler.mode=iddpm_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.1 gpu=0 add_exp_name=anneal-0.9 num_runs=4
python posterior_sample.py +data=ffhq_grayscale +task=phase_retrieval +model=edm_unet_adm_gray_ffhq +sampler=pnp_edm \
       sampler.mode=edm_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.1 gpu=0 add_exp_name=anneal-0.9 num_runs=4

# Black hole
python posterior_sample.py +data=blackhole +task=blackhole_imaging +model=edm_unet_adm_blackhole +sampler=pnp_edm_batch_bh \
       sampler.num_iters=200 sampler.rho=10 sampler.rho_decay_rate=0.93 sampler.rho_min=0.02 sampler.batch_size=100 gpu=0 add_exp_name=final-sim-10-0.02-0.93-200-1e-5-200

# Black hole (real M87)
python posterior_sample.py +data=blackhole_real +task=blackhole_imaging_realM87 +model=edm_unet_adm_blackhole +sampler=pnp_edm_batch_bh \
       sampler.num_iters=200 sampler.rho=10 sampler.rho_decay_rate=0.93 sampler.rho_min=0.02 sampler.batch_size=100 gpu=0 add_exp_name=final-realM87-10-0.02-0.93-200-1e-5-200
