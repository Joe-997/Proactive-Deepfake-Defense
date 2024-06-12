import numpy as np
import torch
import torch.nn as nn


class LinfPGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.01, k=10, a=0.01, b=0.5, feat=None):
        """
        FGSM, I-FGSM and PGD attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.b = b
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device

        # Feature-level attack? Which layer?
        self.feat = feat

        # PGD or I-FGSM?
        self.rand = True
        self.epsilon_small = 0.005
        self.epsilon_large = 0.01

    def perturb_org(self, X_nat, y, c_trg):  # 原图   生成的Deepfake   标签
        """
        Vanilla Attack.
        """

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(f"{name} does not require grad")

        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(
                np.random.uniform(-0.1, 0.1, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()

        for i in range(self.k):
            X.requires_grad = True
            output, feats = self.model(X, c_trg)
            # 通过验证，模型在同一次创建初始化后，多次调用，生成的output是一样的

            # 经过预先处理对原图加入少量随机噪声，output 与 y 有些许不同，但都是原图和标签输入G而得到的结果

            if self.feat:
                output = feats[self.feat]
               #根据测试，这部分完全没被运行过
                # print(output)
            # If a specific feature map is targeted (self.feat is not None), the output is replaced with the targeted feature map.

            self.model.zero_grad()
            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(output, y)  # 针对特征的损失进行攻击。计算output和y的损失

            loss.backward()
            grad = X.grad
            X_adv = X + self.a * grad.sign()
            X_q = X + self.b * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            eta1 = torch.clamp(X_q - X_nat, min=-0.6, max=0.6)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()
            X1 = torch.clamp(X_nat + eta1, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - X_nat, X1

    def perturb_org1(self, X_nat, y, c_trg):  # 原图   生成的Deepfake   标签
        """
        Vanilla Attack.
        """

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(f"{name} does not require grad")

        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(
                np.random.uniform(-0.1, 0.1, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()


        for i in range(self.k):
            X.requires_grad = True
            output, feats = self.model(X, c_trg)
            # 通过验证，模型在同一次创建初始化后，多次调用，生成的output是一样的

            # 经过预先处理对原图加入少量随机噪声，output 与 y 有些许不同，但都是原图和标签输入G而得到的结果


            if self.feat:
                output = feats[self.feat]
               #根据测试，这部分完全没被运行过

                # print(output)
            # If a specific feature map is targeted (self.feat is not None), the output is replaced with the targeted feature map.

            self.model.zero_grad()
            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(output, y)  # 针对特征的损失进行攻击。计算output和y的损失

            loss.backward()
            grad = X.grad

            t = "pgd_dark"

            # X_adv = X + self.a * grad.sign()
            epsilon_map = self.compute_epsilon_map(X, grad, t)

            X_adv = X + epsilon_map * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - X_nat

    def perturb(self, X_nat, y, c_trg):  # 原图   生成的Deepfake   标签
        """
        Vanilla Attack.
        """

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(f"{name} does not require grad")

        if self.rand:
            # X = X_nat.clone() + torch.tensor(
                # np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
            X = X_nat.clone().detach_() + torch.tensor(
                np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            # X = X_nat.clone()
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()

        for i in range(self.k):
            X.requires_grad = True
            output, feats = self.model(X, c_trg)
            # 通过验证，模型在同一次创建初始化后，多次调用，生成的output是一样的

            # 经过预先处理对原图加入少量随机噪声，output 与 y 有些许不同，但都是原图和标签输入G而得到的结果


            if self.feat:
                output = feats[self.feat]
               #根据测试，这部分完全没被运行过

                # print(output)
            # If a specific feature map is targeted (self.feat is not None), the output is replaced with the targeted feature map.

            self.model.zero_grad()
            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(output, y)  # 针对特征的损失进行攻击。计算output和y的损失

            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - X_nat

    def fgsm_attack(self, X_nat, y, c_trg):
        """
        FGSM Attack.
        """
        X = X_nat.clone().detach_() + torch.tensor(
                np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)

        X.requires_grad = True
        output, feats = self.model(X, c_trg)

        if self.feat:
            output = feats[self.feat]

        self.model.zero_grad()
        loss = self.loss_fn(output, y)
        loss.backward()
        grad = X.grad

        t = "fgsm_edge"

        epsilon_map = self.compute_epsilon_map(X, grad, t)

        X_adv = X + epsilon_map * grad.sign()

        X_adv = torch.clamp(X_adv, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X_adv, X_adv - X_nat

    def compute_epsilon_map(self, image_tensor, gradient_tensor, token):
        """
        Compute spatially varying epsilon based on the darkness or lightness of each pixel.
        """
        # Assuming image_tensor has shape (1, 3, H, W)
        grayscale_image = torch.mean(image_tensor, dim=1, keepdim=True)

        # Normalize pixel values to be in [0, 1]
        normalized_image = (grayscale_image + 1) / 2.0

        if token == "fgsm_edge":
            gradient_magnitude = torch.sqrt(torch.sum(gradient_tensor ** 2, dim=1, keepdim=True))
            edge_map = (gradient_magnitude > 0.3).float()
            epsilon_map = torch.where(edge_map > 0, self.epsilon_large, self.epsilon_small)
            return epsilon_map
        elif token == "pgd_dark":
            epsilon_map = torch.where(normalized_image > 0.5, self.epsilon_small, self.epsilon_large)
            return epsilon_map

        # 越大越亮
        # epsilon_map = torch.where(normalized_image > 0.2, self.epsilon_small, self.epsilon_large)
        # epsilon_map = torch.where(edge_map > 0, self.epsilon_large, self.epsilon_small)  # epsilon_map

        # return epsilon_map

        '''
        1.2日记录
        针对第一次汇报：1方法总结，2测试参数解释，3测试结果展示
        
        '''

