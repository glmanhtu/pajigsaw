# ------------
# Adapted from https://github.com/hila-chefer/Transformer-MM-Explainability
# -------------


import argparse
import colorsys
import logging
import os
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms
from PIL import Image
from matplotlib import pyplot as plt

from config import get_config
from data.transforms import TwoImgSyncEval, UnNormalize
from models import build_model
from misc.utils import load_pretrained


def parse_option():
    parser = argparse.ArgumentParser('Pajigsaw visualising script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--pretrained', required=True, help='pretrained weight from checkpoint')
    parser.add_argument('--images', type=str, required=True, nargs='+', help='Path to the two testing images')
    parser.add_argument('--output_dir', type=str, default='visualisation', help='Visualisation output dir')
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument("--threshold", type=float, default=0.6, help="""We visualize masks
            obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument('--tag', help='tag of experiment')

    args, unparsed = parser.parse_known_args()
    args.keep_attn = True

    return args, get_config(args)


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    eye = torch.eye(num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    matrices_aug = all_layer_matrices
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].matmul(joint_attention)
    return joint_attention


# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


# rules 6 + 7 from paper
def apply_self_attention_rules(R_ss, R_sq, cam_ss):
    R_sq_addition = torch.matmul(cam_ss, R_sq)
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition, R_sq_addition


# rule 10 from paper
def apply_mm_attention_rules(R_ss, R_qq, cam_sq, apply_normalization=True, apply_self_in_rule_10=True):
    R_ss_normalized = R_ss
    R_qq_normalized = R_qq
    if apply_normalization:
        R_ss_normalized = handle_residual(R_ss)
        R_qq_normalized = handle_residual(R_qq)
    R_sq_addition = torch.matmul(R_ss_normalized.t(), torch.matmul(cam_sq, R_qq_normalized))
    if not apply_self_in_rule_10:
        R_sq_addition = cam_sq
    R_sq_addition[torch.isnan(R_sq_addition)] = 0
    return R_sq_addition


# normalization- eq. 8+9
def handle_residual(orig_self_attention):
    self_attention = orig_self_attention.clone()
    diag_idx = range(self_attention.shape[-1])
    self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
    assert self_attention[diag_idx, diag_idx].min() >= 0
    self_attention = self_attention / self_attention.sum(dim=-1, keepdim=True)
    self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
    return self_attention


class Generator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_transformer_att(self, img, target_index, index=None):
        outputs = self.model(img)
        kwargs = {"alpha": 1,
                  "target_index": target_index}

        if index == None:
            index = outputs['pred_logits'][0, target_index, :-1].max(1)[1]

        kwargs["target_class"] = index

        one_hot = torch.zeros_like(outputs['pred_logits']).to(outputs['pred_logits'].device)
        one_hot[0, target_index, index] = 1
        one_hot_vector = one_hot.clone().detach()
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * outputs['pred_logits'])

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        self.model.relprop(one_hot_vector, **kwargs)

        decoder_blocks = self.model.transformer.decoder.layers
        encoder_blocks = self.model.transformer.encoder.layers

        # initialize relevancy matrices
        image_bboxes = encoder_blocks[0].attn.get_attn().shape[-1]
        queries_num = decoder_blocks[0].attn.get_attn().shape[-1]

        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(encoder_blocks[0].attn.get_attn().device)
        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(encoder_blocks[0].attn.get_attn().device)
        # impact of image boxes on queries
        self.R_q_i = torch.zeros(queries_num, image_bboxes).to(encoder_blocks[0].attn.get_attn().device)

        # R_q_i generated from last layer
        decoder_last = decoder_blocks[-1]
        cam_q_i = decoder_last.multihead_attn.get_attn_cam().detach()
        grad_q_i = decoder_last.multihead_attn.get_attn_gradients().detach()
        cam_q_i = avg_heads(cam_q_i, grad_q_i)
        self.R_q_i = cam_q_i
        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:, target_index, :].unsqueeze_(0)
        return aggregated

    def handle_self_attention_image(self, blocks):
        for blk in blocks:
            grad = blk.attn.get_attn_gradients().detach()
            if self.use_lrp:
                cam = blk.attn.get_attn_cam().detach()
            else:
                cam = blk.attn.get_attn().detach()
            cam = avg_heads(cam, grad)
            self.R_i_i += torch.matmul(cam, self.R_i_i)

    def handle_co_attn_self_query(self, block):
        grad = block.attn.get_attn_gradients().detach()
        if self.use_lrp:
            cam = block.attn.get_attn_cam().detach()
        else:
            cam = block.attn.get_attn().detach()
        cam = avg_heads(cam, grad)
        R_q_q_add, R_q_i_add = apply_self_attention_rules(self.R_q_q, self.R_q_i, cam)
        self.R_q_q += R_q_q_add
        self.R_q_i += R_q_i_add

    def handle_co_attn_query(self, block):
        if self.use_lrp:
            cam_q_i = block.cross_attn.get_attn_cam().detach()
        else:
            cam_q_i = block.cross_attn.get_attn().detach()
        grad_q_i = block.cross_attn.get_attn_gradients().detach()
        cam_q_i = avg_heads(cam_q_i, grad_q_i)
        self.R_q_i += apply_mm_attention_rules(self.R_q_q, self.R_i_i, cam_q_i,
                                               apply_normalization=self.normalize_self_attention,
                                               apply_self_in_rule_10=self.apply_self_in_rule_10)

    def generate_ours(self, img, target_index, index=None, use_lrp=False, normalize_self_attention=True, apply_self_in_rule_10=True):
        self.use_lrp = use_lrp
        self.normalize_self_attention = normalize_self_attention
        self.apply_self_in_rule_10 = apply_self_in_rule_10
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = self.model(img[None])

        if index == None:
            index = np.argmax(outputs.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, outputs.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * outputs)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        decoder_blocks = self.model.cross_blocks
        encoder_blocks = self.model.blocks

        # initialize relevancy matrices
        image_bboxes = encoder_blocks[0].attn.get_attn().shape[-1]
        queries_num = decoder_blocks[0].attn.get_attn().shape[-1]

        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(encoder_blocks[0].attn.get_attn().device)
        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(encoder_blocks[0].attn.get_attn().device)
        # impact of image boxes on queries
        self.R_q_i = torch.zeros(queries_num, image_bboxes).to(encoder_blocks[0].attn.get_attn().device)

        # image self attention in the encoder
        self.handle_self_attention_image(encoder_blocks)

        # decoder self attention of queries followd by multi-modal attention
        for blk in decoder_blocks:
            # decoder self attention
            self.handle_co_attn_self_query(blk)

            # encoder decoder attention
            self.handle_co_attn_query(blk)
        aggregated = self.R_q_i.unsqueeze_(0)

        return aggregated[0, 1:, :]

    def generate_raw_attn(self, img, target_index):
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = self.model(img[None])

        # get cross attn cam from last decoder layer
        cam_q_i = self.model.cross_blocks[-1].cross_attn.get_attn().detach()
        cam_q_i = cam_q_i.reshape(-1, cam_q_i.shape[-2], cam_q_i.shape[-1])
        cam_q_i = cam_q_i.mean(dim=0)
        self.R_q_i = cam_q_i

        aggregated = self.R_q_i.unsqueeze_(0)

        return aggregated[0, 0, :]

    def generate_rollout(self, img, target_index):
        outputs = self.model(img)

        decoder_blocks = self.model.transformer.decoder.layers
        encoder_blocks = self.model.transformer.encoder.layers

        cams_image = []
        cams_queries = []
        # image self attention in the encoder
        for blk in encoder_blocks:
            cam = blk.attn.get_attn().detach()
            cam = cam.mean(dim=0)
            cams_image.append(cam)

        # decoder self attention of queries
        for blk in decoder_blocks:
            # decoder self attention
            cam = blk.attn.get_attn().detach()
            cam = cam.mean(dim=0)
            cams_queries.append(cam)

        # rollout for self-attention values
        self.R_i_i = compute_rollout_attention(cams_image)
        self.R_q_q = compute_rollout_attention(cams_queries)

        decoder_last = decoder_blocks[-1]
        cam_q_i = decoder_last.multihead_attn.get_attn().detach()
        cam_q_i = cam_q_i.reshape(-1, cam_q_i.shape[-2], cam_q_i.shape[-1])
        cam_q_i = cam_q_i.mean(dim=0)
        self.R_q_i = torch.matmul(self.R_q_q.t(), torch.matmul(cam_q_i, self.R_i_i))
        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:, target_index, :].unsqueeze_(0)
        return aggregated

    def gradcam(self, cam, grad):
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        return cam

    def generate_attn_gradcam(self, img, target_index, index=None):
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = self.model(img[None])

        if index == None:
            index = np.argmax(outputs.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, outputs.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * outputs)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        decoder_blocks = self.model.cross_blocks

        # get cross attn cam from last decoder layer
        cam_q_i = decoder_blocks[-1].cross_attn.get_attn().detach()
        grad_q_i = decoder_blocks[-1].cross_attn.get_attn_gradients().detach()
        cam_q_i = self.gradcam(cam_q_i, grad_q_i)
        self.R_q_i = cam_q_i
        aggregated = self.R_q_i.unsqueeze_(0)

        return aggregated[0, 0, :]


def show_cam_on_image(img, mask, target_size):
    heatmap = cv2.resize(mask, (target_size, target_size))
    img = cv2.resize(img, (target_size, target_size))
    img = np.float32(img) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)



def main(config):
    model = build_model(config).cuda()

    if os.path.isfile(config.MODEL.PRETRAINED):
        load_pretrained(config, model, logger)
    else:
        raise Exception(f'Pretrained model is not exists {config.MODEL.PRETRAINED}')

    model = model.eval()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(config.DATA.IMG_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    un_normaliser = torchvision.transforms.Compose([
        UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        torchvision.transforms.ToPILImage(),
    ])
    assert len(args.images) == 2
    imgs = []
    for img_path in args.images:
        with Image.open(img_path) as f:
            imgs.append(f.convert('RGB'))
    first = transform(imgs[0])
    second = transform(imgs[1])
    images = torch.stack([first, second], dim=0).cuda()

    gen = Generator(model)

    cam = gen.generate_ours(images, target_index=0)

    w_featmap = images.shape[-2] // config.MODEL.PJS.PATCH_SIZE
    h_featmap = images.shape[-1] // config.MODEL.PJS.PATCH_SIZE

    # cam = cam.reshape(1, 1, w_featmap, h_featmap)
    # image_relevance = torch.nn.functional.interpolate(cam, size=images.shape[-2], mode='bilinear')
    # image_relevance = image_relevance.reshape(images.shape[-2], images.shape[-2]).data.cpu().numpy()
    # image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    #
    # image = cv2.resize(np.array(imgs[1]), (images.shape[-2], images.shape[-2]))
    # image = (image - image.min()) / (image.max() - image.min())
    # vis = show_cam_on_image(image, image_relevance)
    # vis = np.uint8(255 * vis)
    # vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    attentions = cam
    colours = random_colors(attentions.shape[1])

    attn_x1_img = np.zeros([w_featmap, h_featmap, 3], dtype=np.float32)
    attn_x2_img = np.zeros([w_featmap, h_featmap, 3], dtype=np.float32)

    x2_feat_img = np.arange(attentions.shape[1]).reshape(w_featmap, h_featmap)
    for h in range(len(x2_feat_img)):
        for w in range(len(x2_feat_img[h])):
            x2_feat_point = h * len(x2_feat_img) + w
            attention_x1 = attentions[x2_feat_point, :].reshape(w_featmap, h_featmap)
            if not torch.all(torch.le(attention_x1, args.threshold)):
                colour = colours[x2_feat_point]
                attn_x2_img[h][w] = colour
                attn_x1_candidates = torch.gt(attention_x1, args.threshold)
                attn_x1_img[attn_x1_candidates.cpu().numpy()] = colour

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(show_cam_on_image(np.array(un_normaliser(first)), attn_x1_img, config.DATA.IMG_SIZE))
    axs[0].axis('off')
    axs[1].imshow(show_cam_on_image(np.array(un_normaliser(second)), attn_x2_img, config.DATA.IMG_SIZE))
    axs[1].axis('off')

    plt.show()


if __name__ == '__main__':
    args, config = parse_option()

    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = logging.getLogger(f"{config.MODEL.NAME}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    main(config)
