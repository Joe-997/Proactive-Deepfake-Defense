from model import Generator
from model import Discriminator
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import attacks
from torchvision import transforms, utils

np.random.seed(0)


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        # self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        # self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        # if self.use_tensorboard:
        # self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            # self.G = AvgBlurGenerator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            #self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim + self.c2_dim + 2, self.g_repeat_num)  # 2 for mask vector.
            # self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim + self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        # self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        # self.print_network(self.D, 'D')

        self.G.to(self.device)
        # self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        # D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))

        # self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.load_model_weights(self.G, G_path)
        # self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def load_model_weights(self, model, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'preprocessing' not in k}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(pretrained_dict, strict=False)

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def test(self, c, test_img):
        """Translate images using StarGAN trained on a single dataset. No attack."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        # Set data loader.
        data_loader = self.celeba_loader

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                # x_real = x_real.to(self.device)
                x_real = test_img.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake, _ = self.G(x_real, c_trg)
                    x_fake_list.append(x_fake)
                    # 生成第一个就中断，原图和黑发图
                    if i == 0:
                        break

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                # result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i + 1))
                result_path = os.path.join(self.result_dir, f'{c}_test_images.jpg')
                # save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                utils.save_image(
                    x_concat[0].unsqueeze(0),
                    result_path,
                    nrow=1,
                    normalize=True,
                )
                print('Saved real and fake images into {}...'.format(result_path))

    # 这个函数是合并代码中使用的
    def test_attack(self, c, test_att_img):
        """Vanilla or blur attacks."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        # Set data loader.
        data_loader = self.celeba_loader

        # Initialize Metrics
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0

        for i, (x_real, c_org) in enumerate(data_loader):
            # Prepare input images and target domain labels.
            # x_real = x_real.to(self.device)
            # img = test_att_img.to(self.device)
            x_real = test_att_img
            x_real = x_real.to(self.device)

            c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

            pgd_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)

            # Translated images.
            x_fake_list = [x_real]   # 第一个
            # x_fake_list = [img]

            for idx, c_trg in enumerate(c_trg_list):
                # 这里控制了只对第一个标签进行攻击并保存结果
                if idx == 1:
                    break
                print('image', i, 'class', idx)
                with torch.no_grad():
                    # x_real_mod = x_real  # 原图
                    x_real_mod = x_real

                    # x_real_mod = self.blur_tensor(x_real_mod) # use blur
                    gen_noattack, gen_noattack_feats = self.G(x_real_mod, c_trg)  # 将原图和标签送入G
                    # 得到生成的图像和特征图列表

                # Attacks
                # x_adv, perturb = pgd_attack.perturb(x_real, gen_noattack, c_trg)  # Vanilla attack
                x_adv, perturb = pgd_attack.perturb(x_real, gen_noattack, c_trg)

                # x_adv, perturb, blurred_image = pgd_attack.perturb_blur(x_real, gen_noattack, c_trg)    # White-box attack on blur
                # x_adv, perturb = pgd_attack.perturb_blur_iter_full(x_real, gen_noattack, c_trg)         # Spread-spectrum attack on blur
                # x_adv, perturb = pgd_attack.perturb_blur_eot(x_real, gen_noattack, c_trg)               # EoT blur adaptation
                '''
                # Generate adversarial example
                x_adv = x_real + perturb
                solver_result = x_adv[0]
                transform = transforms.ToPILImage()
                # 将张量转换为图像
                image1 = transform(solver_result)
                # 保存图像
                image1.save("sample1.jpg")
                '''
                # No attack
                # x_adv = x_real

                # x_adv = self.blur_tensor(x_adv)   # use blur

                # Metrics
                with torch.no_grad():
                    gen, _ = self.G(x_adv, c_trg)

                    # Add to lists
                    # x_fake_list.append(blurred_image)
                    x_fake_list.append(x_adv)   # 第二个
                    # x_fake_list.append(perturb)
                    x_fake_list.append(gen)      # 第三个

                    l1_error += F.l1_loss(gen, gen_noattack)
                    l2_error += F.mse_loss(gen, gen_noattack)
                    l0_error += (gen - gen_noattack).norm(0)
                    min_dist += (gen - gen_noattack).norm(float('-inf'))
                    if F.mse_loss(gen, gen_noattack) > 0.05:
                        n_dist += 1
                    n_samples += 1

            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            # result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i + 1))
            # save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            result_path = os.path.join(self.result_dir, f'{c}_test_att_images.jpg')
            # save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            utils.save_image(
                x_concat[0].unsqueeze(0),
                result_path,
                nrow=1,
                normalize=True,
            )
            if i == 0:  # stop after this many images
                break


        # Print metrics
        print('{} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples,
                                                                                                             l1_error / n_samples,
                                                                                                             l2_error / n_samples,
                                                                                                             float(
                                                                                                                 n_dist) / n_samples,
                                                                                                             l0_error / n_samples,
                                                                                                             min_dist / n_samples))

    # 这个代码用来单独调试
    def test_attack1(self):
        """Vanilla or blur attacks."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        # Set data loader.
        data_loader = self.celeba_loader

        # Initialize Metrics
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0

        for i, (x_real, c_org) in enumerate(data_loader):
            # Prepare input images and target domain labels.
            x_real = x_real.to(self.device)
            # x_real1 = img.to(self.device)
            c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

            pgd_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)

            # Translated images.
            x_fake_list = [x_real]

            for idx, c_trg in enumerate(c_trg_list):
                # 这里控制了只对第一个标签进行攻击并保存结果
                if idx == 1:
                    break
                print('image', i, 'class', idx)
                with torch.no_grad():
                    x_real_mod = x_real  # 原图
                    # x_real_mod = self.blur_tensor(x_real_mod) # use blur
                    gen_noattack, gen_noattack_feats = self.G(x_real_mod, c_trg)  # 将原图和标签送入G
                    # 得到生成的图像和特征图列表
                # Attacks
                x_adv, perturb = pgd_attack.perturb(x_real, gen_noattack, c_trg)  # Vanilla attack
                # x_adv, perturb, blurred_image = pgd_attack.perturb_blur(x_real, gen_noattack, c_trg)    # White-box attack on blur
                # x_adv, perturb = pgd_attack.perturb_blur_iter_full(x_real, gen_noattack, c_trg)         # Spread-spectrum attack on blur
                # x_adv, perturb = pgd_attack.perturb_blur_eot(x_real, gen_noattack, c_trg)               # EoT blur adaptation
                '''
                # Generate adversarial example
                x_adv = x_real + perturb
                solver_result = x_adv[0]
                transform = transforms.ToPILImage()
                # 将张量转换为图像
                image1 = transform(solver_result)
                # 保存图像
                image1.save("sample1.jpg")
                '''
                # No attack
                # x_adv = x_real

                # x_adv = self.blur_tensor(x_adv)   # use blur

                # Metrics
                with torch.no_grad():
                    gen, _ = self.G(x_adv, c_trg)

                    # Add to lists
                    # x_fake_list.append(blurred_image)
                    x_fake_list.append(x_adv)
                    # x_fake_list.append(perturb)
                    x_fake_list.append(gen)

                    l1_error += F.l1_loss(gen, gen_noattack)
                    l2_error += F.mse_loss(gen, gen_noattack)
                    l0_error += (gen - gen_noattack).norm(0)
                    min_dist += (gen - gen_noattack).norm(float('-inf'))
                    if F.mse_loss(gen, gen_noattack) > 0.05:
                        n_dist += 1
                    n_samples += 1

            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i + 1))
            save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            if i == 0:  # stop after this many images
                break


