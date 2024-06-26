import argparse
import os
import torch.distributed as dist
import dist_util
import logger
from script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from image_datasets import load_data
from torchvision import utils
from resizer import Resizer
import math


# added
def load_reference(data_dir, batch_size, image_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["ref_img"] = large_batch
        yield model_kwargs


def main(x, tt):
    args = create_argparser().parse_args()

    # th.manual_seed(0)

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")#, strict=False
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating resizers...")
    assert math.log(args.down_N, 2).is_integer()

    shape = (args.batch_size, 3, args.image_size, args.image_size)
    shape_d = (args.batch_size, 3, int(args.image_size / args.down_N), int(args.image_size / args.down_N))
    down = Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device)
    up = Resizer(shape_d, args.down_N).to(next(model.parameters()).device)
    resizers = (down, up)

    logger.log("loading data...")
    data = load_reference(
        args.base_samples,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("creating samples...")
    count = 0
    while count * args.batch_size < args.num_samples:
        # loader, config = in_gan()
        # print(config)
        # model_kwargs = {}
        '''
        for i, (x_real, c_org) in enumerate(loader):
            # Prepare input images and target domain labels.
            model_kwargs = {'ref_img': x_real}
            break
        '''
        model_kwargs = {'ref_img': x}

        # model_kwargs基本就是原图，只是文件大小略小, 字典model_kwargs["ref_img"]
        # model_kwargs = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        # gan_solver(loader, config)
        # q: 这个sample是什么形式的变量？
        # a: sample是一个tensor，shape为(1, 3, 256, 256)
        sample, denoise, step10 = diffusion.p_sample_loop(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            resizers=resizers,
            range_t=args.range_t,
            ttt=tt
            #loader=loader,
            #config=config
        )
        # print("这里输出i的结果: ", denoise)

        # x_ilvr = sample
        for i in range(args.batch_size):
            out_path = os.path.join(logger.get_dir(),
                                    f"{str(count * args.batch_size + i).zfill(5)}.jpg")
            utils.save_image(
                sample[i].unsqueeze(0),
                out_path,
                nrow=1,
                normalize=True,
                # range=(-1, 1),
            )

        count += 1
        logger.log(f"created {count * args.batch_size} samples")

    dist.barrier()
    logger.log("sampling complete")
    return sample, denoise, step10


def create_argparser():
    '''
        attention_resolutions="16",
        class_cond=False,
        diffusion_steps="1000",
        dropout="0.0",
        image_size="256",
        learn_sigma=True,
        noise_schedule="linear",
        num_channels="128",
        num_head_channels="64",
        num_res_blocks="2",
        resblock_updown=True,
        use_fp16=False,
        use_scale_shift_norm=True,
        timestep_respacing="100",

0.0
--image_size
256
--learn_sigma
True
--noise_schedule
linear
--num_channels
128
--num_head_channels
64
--num_res_blocks
1
--resblock_updown
True
--use_fp16
False
--use_scale_shift_norm
True
--timestep_respacing
100
--model_path
models/celebahq_p2.pt
--base_samples
ref_imgs
--down_N
8
--range_t
0
--save_dir
output




    :return:
    '''
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        down_N=4,   # 4
        range_t=8,   # 0
        use_ddim=False,
        base_samples="ref_imgs",
        model_path="checkpoints/celebahq_p2.pt",
        save_dir="output",
        save_latents=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser





if __name__ == "__main__":
    main()