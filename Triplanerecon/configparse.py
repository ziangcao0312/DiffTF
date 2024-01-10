def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./3D_generation/Checkpoint_all', 
                        help='where to store ckpts and logs')
    parser.add_argument("--ddp", type=int, default=0, 
                        help='whether use ddp')
    parser.add_argument("--idx", type=int, default=0, 
                        help='split_idx')
    parser.add_argument("--num_gpu", type=int, default=1, 
                        help='number of gpus')
    parser.add_argument("--datadir", type=str, default='', 
                        help='input data directory')
    parser.add_argument("--decoderdir", type=str, default='', 
                        help='path of the share decoder')
    parser.add_argument("--state", type=str, default='train', 
                        help='train_mode')
    parser.add_argument("--load_weight", type=int, default=0, 
                        help='whether to load pre-trained model')
    parser.add_argument("--mesh_mode", type=str, default='.obj', 
                        help='format of the save files')
    parser.add_argument("--mesh", type=int, default=0, 
                        help='whether to save mesh files')
    parser.add_argument("--mesh_color", type=int, default=0, 
                        help='whether to save mesh files with texture')
    parser.add_argument("--shape_res", type=int, default=128, 
                        help='resolution of the generated mesh')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2, 
                        help='training batch_size')
    parser.add_argument("--triplanesize", type=int, default=256, 
                        help='the resolution of triplanes')
    parser.add_argument("--num_worker", type=int, default=0, 
                        help='num_worker of dataloader')
    parser.add_argument("--N_iters", type=int, default=200000 + 1, 
                        help='the number of iteration during training')
    parser.add_argument("--train_test", type=float, default=0.9, 
                        help='skip images during test')

    parser.add_argument("--pos_rate", type=float, default=0.5, 
                        help='the ratio of positive area')
    # training options
    parser.add_argument("--using_depth", type=int, default=0, 
                        help='whether to use depth image')
    parser.add_argument("--num_instance", type=int, default=1, 
                        help='number of objects for training')
    parser.add_argument("--triplanechannel", type=int, default=18, 
                        help='the channel of triplanes')
    parser.add_argument("--testvideo", type=int, default=0, 
                        help='whether to save video')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--box_warp", type=float, default=1, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--dataset", type=str, default='omni', 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--triplanepath", type=str, default='/nvme/caoziang/3D_generation/Checkpoint_all/blender_paper_applebatch/ddpm/samples_1x96x256x256.npz', 
                        help='number of coarse samples per ray')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=1, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser