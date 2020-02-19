import torch
import torch.nn as nn
import torch.nn.functional as F
import cma_es
from tensorboardX import SummaryWriter
import config
from torch.distributions import Categorical
from torchvision import transforms
import torchvision.transforms.functional as TVF

from utils import UniImageViewer, plot_keypoints_on_image
from keypoints.models import transporter, functional as KF
from keypoints.ds import datasets as ds
import gym
import gym_wrappers
import torch.multiprocessing as mp
import numpy as np
import iic.models.mnn
import iic.models.classifier
from iic.models.layerbuilder import LayerMetaData
from cma.cma_es import NaiveCovarianceMatrixAdaptation, SimpleCovarianceMatrixAdaptation, FastCovarianceMatrixAdaptation
from cma.cma_es import get_policy, load_weights
from cma.eval import AtariMpEvaluator, AtariSpEvaluator, nop
from iic.utils.viewer import make_grid
import skimage.measure
import cv2
from time import sleep
from matplotlib import pyplot as plt
import pickle

def crop(s):
    return s[34:168, :]


def shrink(s):
    s = skimage.measure.block_reduce(s, (4, 4, 1), np.max)
    s = cv2.resize(s, dsize=(32, 32), interpolation=cv2.INTER_AREA)
    return s


color_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class Keypoints(nn.Module):
    def __init__(self, transporter_net):
        super().__init__()
        self.transporter_net = transporter_net

    def forward(self, s_t):
        heatmap = self.transporter_net.keypoint(s_t)
        kp = KF.spacial_logsoftmax(heatmap)
        return kp


def patches(s, kp, PH, PW):
    """
    Extracts K patches from the input batch of N images
    :param s: the image (N, C, H, W)
    :param kp: K keypoint co-ordinates in range 0.0 - 1.0 (N, K, 2)
    :param PH: height of the patch to extract (int)
    :param PW: width of the patch to extract (int)
    :return: (N, K, C, PH, Pw) image patches
    """
    N, C, H, W = s.shape
    K = kp.size(1)

    def make_affine(kp):
        eye = torch.eye(2, device=kp.device, dtype=kp.dtype).expand(N * K, 2, 2)
        kp = kp.reshape(N * K, 2, 1)
        kp = kp.flip(1)
        w = torch.tensor([2.0, 2.0], device=kp.device, dtype=kp.dtype).expand(N*K, 2).reshape(N * K, 2, 1)
        b = torch.tensor([-1.0, -1.0], device=kp.device, dtype=kp.dtype).expand(N*K, 2).reshape(N * K, 2, 1)
        kp = kp * w + b
        return torch.cat((eye, kp), dim=2)

    affine = make_affine(kp)
    s = s.view(N, 1, C, H, W).expand(N, K, C, H, W).reshape(N * K, C, H, W)
    grid_affine = F.affine_grid(affine, s.shape, align_corners=False)
    patches = F.grid_sample(s, grid_affine, align_corners=False)

    patches = torch.narrow(patches, dim=2, start=H // 2 - PH // 2, length=PH)
    patches = torch.narrow(patches, dim=3, start=W // 2 - PW // 2, length=PW)
    patches = patches.reshape(N, K, C, PH, PW)

    return patches


def view_kp_gl(s, kp_net, device):
    s_32_32_n_t = color_transform(s).unsqueeze(0).to(device)
    kp = kp_net(s_32_32_n_t)
    return kp


def view_gl(s, kp, classifier, device):
    shrink(s)
    s_c_t = TVF.to_tensor(s).unsqueeze(0).to(device)
    patch = patches(s_c_t, kp, 16, 16)
    N, K, C, H, W = patch.shape
    patch = patch.reshape(N * K, C, H, W)
    categories = classifier(patch)
    # lots of cool things we can do from here...
    # here is where we create the key:value pairs Bengio discussed in his talk to NIPS
    categories = categories.argmax(1)  # for now just argmax
    categories = categories.reshape(N, K)
    return kp, categories, patch


def view(s, kp_net, classifier, device, model_labels=None, invert_y=False):
    s_c = crop(s)
    s_32_32 = shrink(s_c)
    s_32_32_n_t = color_transform(s_32_32).unsqueeze(0).to(device)
    kp = kp_net(s_32_32_n_t)
    s_c_t = TVF.to_tensor(s_c).unsqueeze(0).to(device)
    patch = patches(s_c_t, kp, 16, 16)
    N, K, C, H, W = patch.shape
    patch = patch.reshape(N * K, C, H, W)
    categories = classifier(patch)
    # lots of cool things we can do from here...
    # here is where we create the key:value pairs Bengio discussed in his talk to NIPS
    categories = categories.softmax(dim=1)
    categories = categories.reshape(N, K, -1)

    kvp = {}
    if model_labels is not None:
        for k, c in zip(kp[0], categories[0]):
            p = c.max()
            label = model_labels[c.argmax()]
            if label not in kvp:
                kvp[label] = c.max(), k
            elif p > kvp[label][0]:
                kvp[label] = c.max(), k
        z = torch.zeros(1, 3, 3, device=device)
        if 'player' in kvp:
            z[0, 0, 0] = 1.0
            z[0, 0, 1:3] = kvp['player'][1]
        if 'puck' in kvp:
            z[0, 1, 0] = 1.0
            z[0, 1, 1:3] = kvp['puck'][1]
        if 'enemy' in kvp:
            z[0, 2, 0] = 1.0
            z[0, 2, 1:3] = kvp['enemy'][1]
    else:
        z = kp

    if invert_y:
        kp[:, :, 0] = 1 - kp[:, :, 0]
    return z, kp, categories, patch, kvp


def get_action(z, policy, action_map, action_select_mode='argmax'):
    policy_dtype = next(policy.parameters()).dtype
    z = z.type(policy_dtype).flatten()
    p = policy(z)
    if action_select_mode == 'argmax':
        a = torch.argmax(p)
    elif action_select_mode == 'sample':
        a = Categorical(p).sample()
    else:
        a = torch.argmax(p)
    a = action_map(a)
    return a


def evaluate(args, weights, features, depth, render=False, record=False):
    datapack = ds.datasets[args.dataset]
    env = gym.make(datapack.env)
    if args.gym_reward_count_limit is not None:
        env = gym_wrappers.RewardCountLimit(env, args.gym_reward_count_limit)

    actions = datapack.action_map.size
    policy = get_policy(features, actions, depth)
    policy = load_weights(policy, weights)
    policy = policy.to(args.device)

    video = []

    with torch.no_grad():
        v = UniImageViewer(title='full')
        pv = UniImageViewer(title='patch', screen_resolution=(64*3, 64))

        if args.transporter_model_type != 'nop':
            transporter_net = transporter.make(type=args.transporter_model_type,
                                               in_channels=args.transporter_model_in_channels,
                                               z_channels=args.transporter_model_z_channels,
                                               keypoints=args.transporter_model_keypoints,
                                               map_device='cpu',
                                               load=args.transporter_model_load)
            kp_net = Keypoints(transporter_net).to(args.device)

            # init iic categorization model
            encoder, meta = iic.models.mnn.make_layers(args.iic_model_encoder, args.iic_model_type,
                                                       LayerMetaData((3, 16, 16)))
            classifier = iic.models.classifier.Classifier(encoder, meta,
                                                          num_classes=args.iic_model_categories,
                                                          init=args.iic_model_init).to(args.device)

            checkpoint = torch.load(args.iic_model_load)
            classifier.load_state_dict(checkpoint['model'])
        else:
            kp_net = nop

        s = env.reset()
        z, kp, categories, patch, kvp = view(s, kp_net, classifier, args.device, model_labels=args.iic_model_labels)
        a = get_action(z, policy, datapack.action_map, action_select_mode=args.policy_action_select_mode)

        done = False
        reward = 0.0

        while not done:
            s, r, done, i = env.step(a)
            reward += r

            z, kp, categories, patch, kvp = view(s, kp_net, classifier, args.device, model_labels=args.iic_model_labels)
            a = get_action(z, policy, datapack.action_map, action_select_mode=args.policy_action_select_mode)

            record = False
            if render or record:
                if args.transporter_model_keypoints:
                    s = datapack.prepro(s)
                    s = TVF.to_tensor(s).unsqueeze(0)
                    s = plot_keypoints_on_image(kp[0], s[0], radius=4)
                    if render:
                        patch_grid = make_grid(patch, 1, 3)
                        pv.render(patch_grid)
                        # pil = TVF.to_pil_image(patch_grid[0].cpu())
                        # plt.imshow(pil)
                        # plt.title(str([label for label in kvp]))
                        # plt.show()
                        v.render(s)
                        sleep(0.01)
                    if record:
                        video.append(s)
                else:
                    video.append(env.render(mode='rgb_array'))
                    if render:
                        env.render()

    if record and len(video) != 0:
        video = np.expand_dims(np.stack(video), axis=0)
        video = torch.from_numpy(video).permute(0, 1, 4, 2, 3)
        tb.add_video('action replay', video, global_step)

    return reward


if __name__ == '__main__':

    mp.set_start_method('spawn')

    args = config.config()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    datapack = ds.datasets[args.dataset]
    log_dir = f'data/cma_es/{datapack.env}/{args.run_id}/'
    tb = SummaryWriter(log_dir)
    global_step = 0
    best_reward = -1e8
    show = False

    policy_features = args.cma_policy_features

    N = cma_es.parameter_count(get_policy(policy_features, args.cma_policy_depth, datapack.action_map.size))

    evaluator = AtariMpEvaluator(evaluate, args.cma_workers, args, datapack, policy_features, datapack.action_map.size,
                                 args.cma_policy_depth)
    demo = AtariSpEvaluator(evaluate, args, datapack, policy_features, datapack.action_map.size, args.cma_policy_depth,
                            render=args.display, record=True)

    if args.cma_resume is not None:
        with open(args.cma_resume, 'rb') as f:
            cma = pickle.load(f)
    elif args.cma_algo == 'fast':
        cma = FastCovarianceMatrixAdaptation(N=N,
                                             step_mode=args.cma_step_mode,
                                             step_decay=args.cma_step_decay,
                                             initial_step_size=args.cma_initial_step_size,
                                             samples=args.cma_samples,
                                             oversample=args.cma_oversample)
    elif args.cma_algo == 'naive':
        cma = NaiveCovarianceMatrixAdaptation(N=N,
                                              samples=args.cma_samples,
                                              oversample=args.cma_oversample)
    elif args.cma_algo == 'simple':
        cma = SimpleCovarianceMatrixAdaptation(N=N,
                                               samples=args.cma_samples,
                                               oversample=args.cma_oversample)
    else:
        raise Exception('--cma_algo fast | naive | simple')

    tb.add_text('args', str(args), global_step)
    tb.add_text('cma_params', str(cma), global_step)

    for step in cma.recommended_steps:

        ranked_results, info = cma.step(evaluator.fitness)

        print([result['fitness'] for result in ranked_results])

        for key, value in info.items():
            tb.add_scalar(key, value, global_step)

        with open(log_dir + 'checkpoint.pkl', 'wb') as f:
            pickle.dump(cma, f)

        if ranked_results[0]['fitness'] > best_reward:
            best_reward = ranked_results[0]['fitness']
            #torch.save(ranked_results[0]['parameters'], log_dir + 'best_of_generation.pt')
            _p = get_policy(policy_features, datapack.action_map.size, args.cma_policy_depth)
            _p = cma_es.load_weights(_p, weights=ranked_results[0]['parameters'])
            torch.save(_p.state_dict(), log_dir + 'best_policy.pt')
            show = True

        if global_step % args.display_freq == 0 or show:
            demo.fitness(ranked_results[0]['parameters'].unsqueeze(0))

        show = True
        global_step += 1

        if args.epochs is not None and step >= args.epochs:
            break
