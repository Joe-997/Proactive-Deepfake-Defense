import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


# from ilvr_sample import ProjectConfig


def str2bool(v):
    return v.lower() in ('true')


def main_gan(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    celeba_loader = None
    rafd_loader = None

    if config.dataset in ['CelebA', 'Both']:
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)

    return celeba_loader, config


def gan_solver(celeba_loader, config): # , c, img):
    # Solver for training and testing StarGAN.
    solver = Solver(celeba_loader, config)
    # config.mode = 'test'

    solver.test_attack()
    # solver.test1()

    # solver.test_attack1(c)
    # solver.test(c, img)



def in_gan():
    parser_gan = argparse.ArgumentParser()
    # parser_gan = ProjectConfig()

    # Model configuration.
    parser_gan.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser_gan.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser_gan.add_argument('--celeba_crop_size', type=int, default=256, help='crop size for the CelebA dataset')
    parser_gan.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser_gan.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser_gan.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser_gan.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser_gan.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser_gan.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser_gan.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser_gan.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser_gan.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

    # Training configuration.
    parser_gan.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser_gan.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    # 改200000
    parser_gan.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser_gan.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser_gan.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser_gan.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser_gan.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser_gan.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser_gan.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    # parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser_gan.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                            default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    # Test configuration.                              原200000改
    parser_gan.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser_gan.add_argument('--num_workers', type=int, default=1)
    parser_gan.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    # parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    parser_gan.add_argument('--celeba_image_dir', type=str, default='data/celeba/images')
    parser_gan.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser_gan.add_argument('--log_dir', type=str, default='stargan/logs')
    parser_gan.add_argument('--model_save_dir', type=str, default='model_gan')
    parser_gan.add_argument('--sample_dir', type=str, default='samples')
    parser_gan.add_argument('--result_dir', type=str, default='results')

    # Step size.
    parser_gan.add_argument('--log_step', type=int, default=10)
    parser_gan.add_argument('--sample_step', type=int, default=1000)
    parser_gan.add_argument('--model_save_step', type=int, default=5000)
    parser_gan.add_argument('--lr_update_step', type=int, default=1000)

    config = parser_gan.parse_args()
    # print(config)
    # main_gan(config)

    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    celeba_loader = None
    rafd_loader = None

    if config.dataset in ['CelebA', 'Both']:
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)

    return celeba_loader, config

# 以下开始是调试代码


if __name__ == "__main__":

    loader1, fig = in_gan()
    gan_solver(loader1, fig)
