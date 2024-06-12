from ilvr_sample import main
from model import Generator
from PIL import Image
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import attacks
from torchvision import transforms, utils
from torchvision.transforms import v2
from fuliye import gaussian_kernel, apply_inverse_filter

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

    def test1(self):
        """Translate images using StarGAN trained on a single dataset. No attack."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        # Set data loader.
        data_loader = self.celeba_loader

        with torch.no_grad():
            # 读入一张图片并转化为tensor
            j = 0
            for i, (x_real, c_org) in enumerate(data_loader):
                if i == 1:
                    break
                    # Prepare input images and target domain labels.
                    # x_real = test_img.to(self.device)
                    # x_real = x_real.to(self.device)

                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
                x_fake_list = [x_real]                   # 1原图
                for c_trg in c_trg_list:
                    if j == 0:
                        c = c_trg
                        j = 1
                        # x_fake, _ = self.G(x_real, c_trg)
                        # x_fake_list.append(x_fake)
                        # 生成第一个就中断，原图和黑发图
                        #if i == 0:
                    else:
                        break

                x_fake, _ = self.G(x_real, c)
                x_fake_list.append(x_fake)               # 2假图

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i + 1))
            # result_path = os.path.join(self.result_dir, f'{c}_test_images.jpg')
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)

                print('Saved real and fake images into {}...'.format(result_path))

    def test(self, c, test_img):
        """Translate images using StarGAN trained on a single dataset. No attack."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        # Set data loader.
        data_loader = self.celeba_loader

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):
                if i == 0:
                    # Prepare input images and target domain labels.
                    # x_real = x_real.to(self.device)
                    x_real = test_img.to(self.device)
                    c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
                    x_fake_list = [x_real]
                    for c_trg in c_trg_list:
                        x_fake, _ = self.G(x_real, c_trg)
                        x_fake_list.append(x_fake)
                        # 生成第一个就中断，原图和黑发图
                        #if i == 0:
                        break
                else:
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

    def test_attack(self):
        """Vanilla or blur attacks."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        # Set data loader.
        data_loader = self.celeba_loader

        # Initialize Metrics
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0
        tt = None
        kernel_size = 11
        sigma = 0.3
        for i, (x_real, c_org) in enumerate(data_loader):
            # Prepare input images and target domain labels.
            # x_real = x_real.to(self.device)
            # img = test_att_img.to(self.device)
            # x_real = test_att_img
            x_real = x_real.to(self.device)



            x_fake_list = [x_real]                             # 1原图，org_real
            #noise = torch.randn_like(x_real) * 0.02
            #x_real = x_real + noise
            ###############################
            x_real_ilvr, denoise, step10 = main(x_real, tt)    # 原本先从这里获取ilvr结果，现在移动到下面执行
            #noise1 = torch.randn_like(x_real_ilvr) * 0.01
            #x_real_ilvr = x_real_ilvr + noise1

            x_real1 = x_real_ilvr

            """验证denoise
            for change in denoise.values():
                step10 -= change
            step_dir = os.path.join(self.result_dir, 'step10-ilvr-image{}.jpg'.format(i + 1))
            utils.save_image(
                step10[0].unsqueeze(0),
                step_dir,
                nrow=1,
                normalize=True,
            )
            """

            c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
            pgd_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)
            for idx, c_trg in enumerate(c_trg_list):
                # 这里控制了只对第一个标签进行攻击并保存结果
                # 控制只对黑发进行攻击
                if idx == 1:
                    break
                print('image', i, 'class', idx)
                with torch.no_grad():
                    # x_real_mod = x_real
                    #x_real_mod = x_real1
                    # x_real_mod = self.blur_tensor(x_real_mod) # use blur

                    #gen_noattack, gen_noattack_feats = self.G(x_real1, c_trg)  # 将原图和标签送入G
                    # 得到生成的图像和特征图列表
                    x_fake, _ = self.G(x_real, c_trg)
                    x_fake_list.append(x_fake)               # 2原图进入G的结果，org_fake
                    gen_noattack, gen_noattack_feats = self.G(x_real1, c_trg)
                    x_fake_list.append(x_real1)              # 3ilvr结果，ilvr_real
                    x_fake_list.append(gen_noattack)         # 4ilvr结果进入G的结果,ilvr_fake
                """
                x_adv, perturb, x_1 = pgd_attack.perturb_org(x_real, x_fake, c_trg)
                    # 在这里返回一个强的x_1和弱的噪声
                x_real_ilvr_qiang, denoise, step10 = main(x_real, x_1)
                x_real1 = x_real_ilvr_qiang
                """
                # with torch.no_grad():


                    # 3ilvr结果

                result_path_ilvr = os.path.join(self.result_dir, '{}-ilvr-image.jpg'.format(i + 1))
                utils.save_image(
                    x_real1[0].unsqueeze(0),
                    result_path_ilvr,
                    nrow=1,
                    normalize=True,
                )
                result_path_ilvr_black = os.path.join(self.result_dir, '{}-ilvr-black.jpg'.format(i + 1))
                utils.save_image(
                    gen_noattack[0].unsqueeze(0),
                    result_path_ilvr_black,
                    nrow=1,
                    normalize=True,
                )

                """
                2/18
                这里新增了denoise，参考perturb_fgsm1来修改denoise_all
                初步实现在最后10step中，对图片插入噪声（将最后十步中的对抗信号保留）
                """


                # 这里fgsm都是0，具体原因需要调查
                #noise = torch.randn_like(x_real) * 0.03
                #x_real = x_real + noise
                x_adv, perturb = pgd_attack.perturb_org1(x_real, x_fake, c_trg)
                x_advv, perturbb, x_1 = pgd_attack.perturb_org(x_real1, gen_noattack, c_trg)
                # 在这里返回一个强的x_1和弱的噪声
                #x_real_ilvr_qiang, denoise, step10 = main(x_real1, tt)
                x_real_ilvr_qiang, denoise, step10 = main(x_real1, x_1)
                x_real_q = x_real_ilvr_qiang

                with torch.no_grad():
                    test_sample, _ = self.G(x_real_q, c_trg)
                    x_fake_list.append(test_sample)                # 5对抗样本，org_adv_sample

                    result_path_ilvr_new = os.path.join(self.result_dir, '{}-ilvr_strong_sample.jpg'.format(i + 1))
                    utils.save_image(
                        x_real_q[0].unsqueeze(0),
                        result_path_ilvr_new,
                        nrow=1,
                        normalize=True,
                    )
                    result_path_ilvr_q = os.path.join(self.result_dir, '{}-ilvr_strong_result.jpg'.format(i + 1))
                    utils.save_image(
                        test_sample[0].unsqueeze(0),
                        result_path_ilvr_q,
                        nrow=1,
                        normalize=True,
                    )
                    result_path_ilvr_qq = os.path.join(self.result_dir, '{}-ilvr_x_1.jpg'.format(i + 1))
                    utils.save_image(
                        x_1[0].unsqueeze(0),
                        result_path_ilvr_qq,
                        nrow=1,
                        normalize=True,
                    )



                # x_fgsm, perturb_fgsm = pgd_attack.fgsm_attack(x_real, x_fake, c_trg)
                #x_adv1, perturb1 = pgd_attack.perturb_org1(x_real1, gen_noattack, c_trg)
                #x_adv1, perturb1 = pgd_attack.perturb_org1(x_real_q, test_sample, c_trg)

                """
                x_fgsm1, perturb_fgsm1 = pgd_attack.fgsm_attack(x_real1, gen_noattack, c_trg)
                denoise_all = torch.zeros(1, 3, 256, 256).to(self.device)
                for change in denoise.values():
                    denoise_all += change
                mask = (perturb_fgsm1 * denoise_all) > 0
                denoise_all[~mask] = perturb_fgsm1[~mask]
                denoise_all[mask] = denoise_all[mask] * 0.05
                
                # x_combine = x_real + (perturb * 0.6 + perturb_fgsm * 0.4)
                x_combine = x_real + perturb
                #x_combine1 = x_real1 + (perturb1 * 0.6 + perturb_fgsm1 * 0.4)
                p = perturb1 * 0.6 + denoise_all * 0.4
                x_combine1 = x_real1 + (perturb1 * 0.7 + denoise_all * 0.3)
                """

                # 创建高斯核
                kernel = gaussian_kernel(kernel_size, sigma)
                # 应用逆滤波器
                restored_perturb_ilvr = apply_inverse_filter(perturbb, kernel, device=self.device)
                restored_perturb_pure = apply_inverse_filter(perturb, kernel, device=self.device)

                # 高斯噪声
                x_combine = x_real + restored_perturb_pure
                x_combine1 = x_real1 + restored_perturb_ilvr

                blurrer = v2.GaussianBlur(kernel_size, sigma)
                g_noise_org = blurrer(x_combine)
                g_noise_ilvr = blurrer(x_combine1)

                #x_combine = g_noise_org
                #x_combine1 = g_noise_ilvr
                x_combine.clamp_(-1, 1)
                x_combine1.clamp_(-1, 1)


                num = torch.sum(perturb.eq(0))
                num1 = torch.sum(perturbb.eq(0))
                print('原噪声0的个数', num, '生成噪声0的个数', num1)

                result_path_sample = os.path.join(self.result_dir, '{}-real-image-sample.jpg'.format(i + 1))
                utils.save_image(
                    x_combine[0].unsqueeze(0),
                    result_path_sample,
                    nrow=1,
                    normalize=True,
                )
                result_path_sample1 = os.path.join(self.result_dir, '{}-ilvr-image-sample.jpg'.format(i + 1))
                utils.save_image(
                    x_combine1[0].unsqueeze(0),
                    result_path_sample1,
                    nrow=1,
                    normalize=True,
                )
                result_path_sample_blur = os.path.join(self.result_dir, '{}-real-image-sample_blur.jpg'.format(i + 1))
                utils.save_image(
                    g_noise_org[0].unsqueeze(0),
                    result_path_sample_blur,
                    nrow=1,
                    normalize=True,
                )
                result_path_sample1_blur = os.path.join(self.result_dir, '{}-ilvr-image-sample_blur.jpg'.format(i + 1))
                utils.save_image(
                    g_noise_ilvr[0].unsqueeze(0),
                    result_path_sample1_blur,
                    nrow=1,
                    normalize=True,
                )

                # Metrics
                with torch.no_grad():
                    # gen, _ = self.G(x_adv, c_trg)
                    gen, _ = self.G(x_combine, c_trg)
                    gen1, _ = self.G(x_combine1, c_trg)
                    gen_b, _ = self.G(g_noise_org, c_trg)
                    gen1_b, _ = self.G(g_noise_ilvr, c_trg)

                    # Add to lists
                    x_fake_list.append(x_combine1)               # 6对抗样本，ilvr+q
                    x_fake_list.append(gen1)                     # 7对抗样本结果,ilvr+q
                    x_fake_list.append(x_combine)                # 8对抗样本，org
                    x_fake_list.append(gen)                      # 9对抗样本，org_fake

                    x_fake_list.append(g_noise_ilvr)              # 10对抗样本，ilvr+q
                    x_fake_list.append(gen1_b)                    # 11对抗样本结果,ilvr+q
                    x_fake_list.append(g_noise_org)               # 12对抗样本，org
                    x_fake_list.append(gen_b)                     # 13对抗样本，org_fake

                    result_path_sample_gen = os.path.join(self.result_dir, '{}-ilvr7-result.jpg'.format(i + 1))
                    utils.save_image(
                        gen1[0].unsqueeze(0),
                        result_path_sample_gen,
                        nrow=1,
                        normalize=True,
                    )
                    result_path_sample_gen1 = os.path.join(self.result_dir, '{}-org9-result.jpg'.format(i + 1))
                    utils.save_image(
                        gen[0].unsqueeze(0),
                        result_path_sample_gen1,
                        nrow=1,
                        normalize=True,
                    )
                    result_path_sample_gen1 = os.path.join(self.result_dir, '{}-org9-sample.jpg'.format(i + 1))
                    utils.save_image(
                        g_noise_ilvr[0].unsqueeze(0),
                        result_path_sample_gen1,
                        nrow=1,
                        normalize=True,
                    )
                    result_path_sample_gen_b = os.path.join(self.result_dir, '{}-ilvr11-result_blur.jpg'.format(i + 1))
                    utils.save_image(
                        gen1_b[0].unsqueeze(0),
                        result_path_sample_gen_b,
                        nrow=1,
                        normalize=True,
                    )
                    result_path_sample_gen1_b = os.path.join(self.result_dir, '{}-org13-result_blur.jpg'.format(i + 1))
                    utils.save_image(
                        gen_b[0].unsqueeze(0),
                        result_path_sample_gen1_b,
                        nrow=1,
                        normalize=True,
                    )

                    l1_error += F.l1_loss(gen1, gen_noattack)
                    l2_error += F.mse_loss(gen1, gen_noattack)
                    l0_error += (gen1 - gen_noattack).norm(0)
                    min_dist += (gen1 - gen_noattack).norm(float('-inf'))
                    if F.mse_loss(gen, gen_noattack) > 0.05:
                        n_dist += 1
                    n_samples += 1

            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i + 1))
            utils.save_image(
                x_concat[0].unsqueeze(0),
                result_path,
                nrow=1,
                normalize=True,
            )


            ################################
            """
            # 插入ILVR,替换x_real,   将x_real传入main,得到step1的结果
            tt = 0
            x_real_ilvr = main(x_real, tt)
            x_real1 = x_real_ilvr
            """
            """
            c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
            pgd_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)
            for idx, c_trg in enumerate(c_trg_list):
                # 这里控制了只对第一个标签进行攻击并保存结果
                # 控制只对黑发进行攻击
                if idx == 1:
                    break
                print('image', i, 'class', idx)
                with torch.no_grad():
                    # x_real_mod = x_real
                    x_real_mod = x_real1
                    # x_real_mod = self.blur_tensor(x_real_mod) # use blur
                    gen_noattack, gen_noattack_feats = self.G(x_real_mod, c_trg)  # 将原图和标签送入G
                    # 得到生成的图像和特征图列表
                    x_fake, _ = self.G(x_real, c_trg)
                    x_fake_list.append(x_fake)                   # 2原图进入G的结果，org_fake
                    x_fake_list.append(x_real1)                  # 3ilvr结果，ilvr_real
                    x_fake_list.append(gen_noattack)             # 4ilvr结果进入G的结果,ilvr_fake

                                      # 3ilvr结果

                result_path_ilvr = os.path.join(self.result_dir, '{}-ilvr-image.jpg'.format(i + 1))
                utils.save_image(
                    x_real1[0].unsqueeze(0),
                    result_path_ilvr,
                    nrow=1,
                    normalize=True,
                )

                x_adv, perturb = pgd_attack.perturb(x_real, x_fake, c_trg)
                x_fgsm, perturb_fgsm = pgd_attack.fgsm_attack(x_real, x_fake, c_trg)
                x_combine = x_real + (perturb * 0.6 + perturb_fgsm * 0.4)

                x_adv1, perturb1 = pgd_attack.perturb(x_real1, gen_noattack, c_trg)
                x_fgsm1, perturb_fgsm1 = pgd_attack.fgsm_attack(x_real1, gen_noattack, c_trg)
                x_combine1 = x_real1 + (perturb1 * 0.6 + perturb_fgsm1 * 0.4)

                result_path_sample = os.path.join(self.result_dir, '{}-real-image-sample.jpg'.format(i + 1))
                utils.save_image(
                    x_combine[0].unsqueeze(0),
                    result_path_sample,
                    nrow=1,
                    normalize=True,
                )
                result_path_sample1 = os.path.join(self.result_dir, '{}-ilvr-image-sample.jpg'.format(i + 1))
                utils.save_image(
                    x_combine1[0].unsqueeze(0),
                    result_path_sample1,
                    nrow=1,
                    normalize=True,
                )

                # Metrics
                with torch.no_grad():
                    # gen, _ = self.G(x_adv, c_trg)
                    gen, _ = self.G(x_adv1, c_trg)
                    # Add to lists
                    # x_fake_list.append(x_adv)                   
                    x_fake_list.append(x_adv1)                  # 5对抗样本，org_adv_sample
                    x_fake_list.append(gen)                     # 6对抗样本结果,org_adv_fake

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
            utils.save_image(
                x_concat[0].unsqueeze(0),
                result_path,
                nrow=1,
                normalize=True,
            )
            """
            if i == 9:  # stop after this many images
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
    def test_attack1(self, c):
        """Vanilla or blur attacks."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        # Set data loader.
        data_loader = self.celeba_loader

        # Initialize Metrics
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0
        # img.requires_grad = True


        img = Image.open('output/diff_temp.jpg')
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])
        # 应用转换
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)

        for i, (x_real, c_org) in enumerate(data_loader):
            # Prepare input images and target domain labels.
            # x_real = x_real.to(self.device)

            x_real = img_tensor.to(self.device)
            # x_real = test_att_img.to(self.device)
            # x_real.requires_grad = True

            c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

            pgd_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)

            # Translated images.
            x_fake_list = [x_real]

            for idx, c_trg in enumerate(c_trg_list):
                # 这里控制了只对第一个标签进行攻击并保存结果
                if idx == 1:
                    break
                print('image', i, 'class', idx)
                x_real_mod = x_real  # 原图
                # x_real_mod = self.blur_tensor(x_real_mod) # use blur
                gen_noattack, gen_noattack_feats = self.G(x_real_mod, c_trg)  # 将原图和标签送入G
                '''
                with torch.no_grad():
                    x_real_mod = x_real  # 原图
                    # x_real_mod = self.blur_tensor(x_real_mod) # use blur
                    gen_noattack, gen_noattack_feats = self.G(x_real_mod, c_trg)  # 将原图和标签送入G
                    # 得到生成的图像和特征图列表
                '''

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
                '''
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
                '''

            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = os.path.join(self.result_dir, f'{c}-images.jpg'.format(i + 1))
            save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            if i == 0:  # stop after this many images
                break


