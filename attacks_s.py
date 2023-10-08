import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

class LinfPGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.01, k=10, a=0.01, feat=None):   # 0.005默认epsilon
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
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device

        # Feature-level attack? Which layer?
        self.feat = feat

        # PGD or I-FGSM?
        self.rand = True

    def perturb(self, X_nat, y, c_trg):  # 原图   生成的Deepfake   标签
        """
        Vanilla Attack.
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(
                np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()

        for i in range(self.k):
            X.requires_grad = True
            output, feats = self.model(X, c_trg)
            '''
            sample1 = output[0]
            sample2 = y[0]
            transform = transforms.ToPILImage()
            # 将张量转换为图像
            image1 = transform(sample1)
            # 保存图像
            image1.save("sample1.jpg")
            image2 = transform(sample2)
            # 保存图像
            image2.save("sample2.jpg")
            print(len(y))
            '''

            if self.feat:
                output = feats[self.feat]
               #根据测试，这部分完全没被运行过

                # print(output)
            # If a specific feature map is targeted (self.feat is not None), the output is replaced with the targeted feature map.

            self.model.zero_grad()
            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(output, y)  # 针对特征的损失进行攻击
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()  # +- has style difference, but still worked

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()
        #sample = X[0]
        #transform = transforms.ToPILImage()
        # 将张量转换为图像
        #image = transform(sample)
        # 保存图像
        #image.save("output_image(after).jpg")

        return X, X - X_nat
