# True False

import os
run_mode = os.environ.get('run_mode')

if run_mode == 'train':
    # train parameter
    debug_d_reg = True
    debug_g_reg = True
    debug_optim = True
    debug_use_op = True

    debug_channels_size = True
    # debug_channels_size = False
    channels_ratio = '11'
    if debug_channels_size:
        # channels_ratio = '12'
        channels_ratio = '14'
        # channels_ratio = 'StyleKD'
        if channels_ratio == '12':
            channel_multiplier = 2
            debug_channels_g = {
                4: 512 // 2,
                8: 512 // 2,
                16: 512 // 2,
                32: 512 // 2,
                64: 256 * channel_multiplier // 2,
                128: 128 * channel_multiplier // 2,
                256: 64 * channel_multiplier // 2,
                512: 32 * channel_multiplier // 2,
                1024: 16 * channel_multiplier // 2,
            }
        elif channels_ratio == '14':
            channel_multiplier = 2
            debug_channels_g = {
                4: 512 // 4,
                8: 512 // 4,
                16: 512 // 4,
                32: 512 // 4,
                64: 256 * channel_multiplier // 4,
                128: 128 * channel_multiplier // 4,
                256: 64 * channel_multiplier // 4,
                512: 32 * channel_multiplier // 4,
                1024: 16 * channel_multiplier // 4,
            }
        elif channels_ratio == 'StyleKD':
            debug_channels_g = {
                4: 154,
                8: 154,
                16: 154,
                32: 154,
                64: 154,
                128: 77,
                256: 39,
                512: 20,
                1024: 10,
            }

    debug_xflip = False

    debug_flops = False

    # use random noise or not
    debug_randomize_noise = True

    debug_hw = '11'  # 11 21 34
    debug_crop_padding = False
    debug_real_color = 'BGR'  # RGB BGR

    debug_real_fixed = False
    debug_noise_fixed = False
    debug_sample_z_fixed = True

    debug_print_param = False
    debug_print_grad = False

    debug_fp16 = False

else:
    # test parameter
    debug_use_op = False

    debug_channels_size = True
    # debug_channels_size = False
    # channels_ratio = '11'
    if debug_channels_size:
        # channels_ratio = '12'
        channels_ratio = '14'
        # channels_ratio = 'StyleKD'
        if channels_ratio == '12':
            channel_multiplier = 2
            debug_channels_g = {
                4: 512//2,
                8: 512//2,
                16: 512//2,
                32: 512//2,
                64: 256 * channel_multiplier//2,
                128: 128 * channel_multiplier//2,
                256: 64 * channel_multiplier//2,
                512: 32 * channel_multiplier//2,
                1024: 16 * channel_multiplier//2,
            }
        elif channels_ratio == '14':
            channel_multiplier = 2
            debug_channels_g = {
                4: 512//4,
                8: 512//4,
                16: 512//4,
                32: 512//4,
                64: 256 * channel_multiplier//4,
                128: 128 * channel_multiplier//4,
                256: 64 * channel_multiplier//4,
                512: 32 * channel_multiplier//4,
                1024: 16 * channel_multiplier//4,
            }
        elif channels_ratio == 'StyleKD':
            debug_channels_g = {
                4: 154,
                8: 154,
                16: 154,
                32: 154,
                64: 154,
                128: 77,
                256: 39,
                512: 20,
                1024: 10,
            }

    debug_flops = False

    # use random noise or not
    debug_randomize_noise = False

    debug_hw = '11'  # 11 21 34
    debug_real_color = 'BGR'  # RGB BGR

    debug_sample_z_fixed = False

    debug_fp16 = False

    debug_save_latent = False
    debug_use_latent = False
