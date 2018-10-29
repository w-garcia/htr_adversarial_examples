import models.crnn
import utils
import dataset

import numpy as np
import torch
import torch.optim as optim
import argparse
import logging
import io
import os
import itertools
import pickle

from warpctc_pytorch import CTCLoss
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from carlini_attack_utils import interpolate, resizeDifferentiableNormalize

logging.basicConfig()
logger = logging.getLogger(__name__)
# Change logging level to info if running experiment, debug otherwise
logger.setLevel(logging.DEBUG)

# SEED = 9
nc = 1
imgH = 40
imgW = 180
num_hidden = 256


class CarliniAttack:
    def __init__(self, oracle, alphabet, image_shape, target, file_weights):
        self.learning_rate = 0.001
        # self.learning_rate = 10
        self.num_iterations = 5000
        # self.num_iterations = 100
        self.batch_size = bs = 1
        self.phrase_length = len(target)
        self.o_imW, self.o_imH = image_shape
        self.i_imW, self.i_imH = imgW, imgH
        self.oracle = oracle
        self.weights = file_weights

        # Variable for adversarial noise, which is added to the image to perturb it
        if torch.cuda.is_available():
            self.delta = Variable(torch.rand((1, self.o_imH, self.o_imW)).cuda(), requires_grad=True)
        else:
            self.delta = Variable(torch.rand((1, self.o_imH, self.o_imW)), requires_grad=True)

        # Optimize on delta and use ctc as criterion
        ctcloss = CTCLoss()
        self.optimizer = optim.Adam([self.delta],
                                    lr=self.learning_rate,
                                    betas=(0.9, 0.999))

        self.loss = ctcloss
        self.ctcloss = ctcloss
        self.target = target
        self.converter = utils.strLabelConverter(alphabet, attention=False)

    def _reset_oracle(self):
        self.oracle.load_state_dict(self.weights)

    def _tensor_to_PIL_im(self, tensor, mode='L'):
        if mode == 'F':
            # The ToPILImage impelementation for 'F' mode is broken
            # https://github.com/pytorch/vision/issues/448
            tensor = tensor.cpu().detach()
            tensor = tensor.mul(255)
            tensor = np.transpose(tensor.numpy(), (1, 2, 0))
            tensor = tensor.squeeze()

            return Image.fromarray(tensor, mode='F')
        else:
            imager = ToPILImage(mode='L')
            pil_im = imager(tensor.cpu().detach())
            return pil_im

    def execute(self, images, out_dir):
        img_path = images[0]
        bs = self.batch_size
        tensorizer = ToTensor()
        transformer = dataset.resizeNormalize((imgW, imgH))

        image = Image.open(img_path).convert('L')
        original_pil = image

        # First just convert to tensor, we need to use a differentiable resize fn later.
        image = tensorizer(image)
        # image = transformer(image)

        if torch.cuda.is_available():
            image = image.cuda()

        image = Variable(image)

        original = image

        # Get optimizable version of target and length
        length = Variable(torch.IntTensor(bs))
        text = Variable(torch.IntTensor(bs * 5))  # ????????????????

        t, l = self.converter.encode(self.target)
        utils.loadData(length, l)
        utils.loadData(text, t)

        # This controls the mask to create around the border of fonts.
        # 1.0 = mask away white pixels. ~0.7 = mask closer to font . 0.0 = mask away nothing
        whitespace_mask = (original < 0.7).to(dtype=torch.float32)

        dc = 0.75
        for i in range(self.num_iterations):
            # if i % 200 == 0 and i != 0:
            #     sofar = self.delta.cpu().detach().numpy() * 255.
            #     sofar = np.round(sofar).astype(dtype=np.int8)
            #     sofar = sofar / 255.
            #     self.delta = Variable(torch.tensor(sofar).float().cuda(), requires_grad=True)
            #
            #     self.optimizer = optim.Adam([self.delta],
            #                                 lr=self.learning_rate,
            #                                 betas=(0.9, 0.999))

            apply_delta = torch.clamp(self.delta, min=-dc, max=dc)
            apply_delta = apply_delta * whitespace_mask

            pass_in = torch.clamp(apply_delta + original, min=0.0, max=1.0)

            # Now we need to quantize our tensor down to 8 bit precision.
            # If we don't, a lot of adversarial info is lost when we go from float32 to int8 (0-255, for PIL).
            # This makes the optim converge slower but is necessary so info isn't lost during conversions
            # pass_in = pass_in.to(dtype=torch.uint8)
            # pass_in = pass_in.to(dtype=torch.float32)
            # if i % 100 == 0 and i != 0:
            #     self.delta = self._tensor_to_PIL_im(self.delta)
            #     self.delta = tensorizer(self.delta)
            #
            #     if torch.cuda.is_available():
            #         self.delta = self.delta.cuda()
            #
            #     self.delta = Variable(self.delta, requires_grad=True)

            # pass_in = pass_in.view(1, *pass_in.size())

            # Pass to differentiable resize
            # This would work better if the model was trained with such an end to end architecture
            pass_in = pass_in.view(1, *pass_in.size())
            pass_in = interpolate(pass_in,
                                  size=(self.i_imH, self.i_imW),
                                  mode='bilinear', align_corners=True)

            # self._tensor_to_PIL_im(pass_in[0]).show()

            # Instead use our own differentiable version of PIL resizer
            # transformer = resizeDifferentiableNormalize((imgW, imgH))
            # image = transformer(new_input)
            # if torch.cuda.is_available():
            #     image = image.cuda()
            #
            # image = image.view(1, *image.size())
            # image = Variable(image)

            logits = self.oracle(pass_in)

            # Model already restored
            preds_size = Variable(torch.IntTensor([logits.size(0)] * bs))

            cost = self.ctcloss(logits, text, preds_size, length) / bs
            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()
            # self._reset_oracle()

            if i % 10 == 0:
                logger.debug("iteration: {}, cost: {}".format(i, cost.data))
            if i % 100 == 0:
                # See how we're doing
                _, sim_pred = decode_logits(logits)
                logger.debug("Decoding at iteration {}: {}".format(i, sim_pred))
                if sim_pred == self.target:
                    # We're done
                    logger.debug("Early stop.")
                    break

        self.oracle.eval()

        _, original_im_pred = classify_image_pil(self.oracle, original_pil)
        logger.debug("Original image classify: {}".format(original_im_pred))

        apply_delta = torch.clamp(self.delta, min=-dc, max=dc)
        apply_delta = apply_delta * whitespace_mask

        pass_in = torch.clamp(apply_delta + original, min=0.0, max=1.0)
        pil_attack_float = self._tensor_to_PIL_im(pass_in, mode='F')
        pil_mask = self._tensor_to_PIL_im(apply_delta, mode='L')
        pil_attack_int = self._tensor_to_PIL_im(pass_in, mode='L')

        _, attack_pil_classify = classify_image_pil(self.oracle, pil_attack_int)
        logger.debug("PIL-based image classify: {}".format(attack_pil_classify))

        pass_in = pass_in.view(1, *pass_in.size())
        pass_in = interpolate(pass_in,
                              size=(self.i_imH, self.i_imW),
                              mode='bilinear', align_corners=True)

        new_attack_input = pass_in

        _, attack_ete_classify = classify_image_tensor(self.oracle, new_attack_input)
        logger.debug("Attacked E-t-E classify: {}".format(attack_ete_classify))

        # original_pil.show()
        # pil_attack_int.show()

        run_id = np.random.randint(999999)
        original_path = os.path.join(out_dir, 'original_{}.jpg'.format(run_id))
        delta_path = os.path.join(out_dir, 'delta_{}.jpg'.format(run_id))
        pil_attack_float_path = os.path.join(out_dir, 'attack_{}.tiff'.format(run_id))
        pil_attack_int_path = os.path.join(out_dir, 'attack_{}.jpg'.format(run_id))
        out_ckpt_path = os.path.join(out_dir, 'CTC-CRNN_{}.pt'.format(run_id))

        original_pil.save(original_path)
        pil_mask.save(delta_path)
        pil_attack_float.save(pil_attack_float_path)
        pil_attack_int.save(pil_attack_int_path)

        torch.save(self.oracle.state_dict(), out_ckpt_path)
        logger.debug("Saved to ID {}".format(run_id))

        pickle.dump((original_im_pred, attack_ete_classify), open(os.path.join(out_dir, 'result.pkl'), 'wb'))


def decode_logits(preds):
    _, preds = preds.max(2)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    preds = preds.transpose(1, 0).contiguous().view(-1)

    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return raw_pred, sim_pred


def image_pil_to_logits(oracle, pil_im):
    transformer = dataset.resizeNormalize((imgW, imgH))
    image = transformer(pil_im)
    if torch.cuda.is_available():
        image = image.cuda()

    image = image.view(1, *image.size())
    image = Variable(image)

    preds = oracle(image)
    return preds


def classify_image_pil(oracle, pil_im):
    preds = image_pil_to_logits(oracle, pil_im)

    _, preds = preds.max(2)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    preds = preds.transpose(1, 0).contiguous().view(-1)

    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return raw_pred, sim_pred


def classify_image_tensor(oracle, tensor):
    # imager = ToPILImage()
    # pil_im = imager(tensor.cpu())

    preds = oracle(tensor)
    _, preds = preds.max(2)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    preds = preds.transpose(1, 0).contiguous().view(-1)

    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    #
    return raw_pred, sim_pred
    #
    # return classify_image_pil(oracle, pil_im)


def _validate(args):
    pass


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='+', required=True, help="Input .jpeg images, seperated by spaces")
    parser.add_argument('--target', type=str, required=True, help="Target transcription for images.")
    parser.add_argument('--out', required=True, help="Directory to save adversarial examples.")
    parser.add_argument('--ckpt', required=True, help="Path to the best trained netCRNN .pth file.")
    parser.add_argument('--alphabet', required=True, help="Path to alphabet of the trained checkpoint.")
    parser.add_argument('--cuda', action='store_true', default=False, help="Use CUDA.")
    parser.add_argument('--seed', default=9, help="Random seed to use.")
    args = parser.parse_args()

    _validate(args)

    torch.manual_seed(args.seed)

    try:
        with io.open(args.alphabet, 'r', encoding='utf-8') as myfile:
            alphabet = myfile.read().split()
            alphabet.append(u' ')
            alphabet = ''.join(alphabet)

        converter = utils.strLabelConverter(alphabet, attention=False)

        nclass = converter.num_classes

        crnn = models.crnn.CRNN(imgH, nc, nclass, num_hidden)
        crnn.apply(weights_init)

        if args.cuda:
            crnn = crnn.cuda()
            crnn = torch.nn.DataParallel(crnn)

        logger.info("Loading pretrained model from {}".format(args.ckpt))
        file_weights = torch.load(args.ckpt)

        crnn.load_state_dict(file_weights)

        print("The oracle network:", crnn)  # Logging can't print torch models :thinking:

        image = Image.open(args.input[0]).convert('L')
        attack = CarliniAttack(crnn, alphabet, image.size, args.target, file_weights)

        attack.execute(args.input, args.out)

    except KeyboardInterrupt:
        pass

