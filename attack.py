import argparse
import time
import numpy as np
import data
import models
import os
import utils
from datetime import datetime
np.set_printoptions(precision=5, suppress=True)

from torchvision import transforms
from PIL import Image
import torch

preprocess = None

def nchw_to_nhwc(x):
    """
    Convert a NCHW tensor to NHWC tensor.
    """
    return torch.tensor(np.array([x.numpy().transpose(1, 2, 0)]))

def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p


def pseudo_gaussian_pert_rectangles(x, y):
    delta = np.zeros([x, y])
    x_c, y_c = x // 2 + 1, y // 2 + 1

    counter2 = [x_c - 1, y_c - 1]
    for counter in range(0, max(x_c, y_c)):
        delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
              max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

        counter2[0] -= 1
        counter2[1] -= 1

    delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def meta_pseudo_gaussian_pert(s):
    delta = np.zeros([s, s])
    n_subsquares = 2
    if n_subsquares == 2:
        delta[:s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s)
        delta[s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        if np.random.rand(1) > 0.5: delta = np.transpose(delta)

    elif n_subsquares == 4:
        delta[:s // 2, :s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, :s // 2] = pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
        delta[:s // 2, s // 2:] = pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def square_attack_l2(model, x, y, corr_classified, eps, n_iters, p_init, metrics_path, targeted, loss_type):
    """ The L2 square attack """
    np.random.seed(0)

    min_val, max_val = 0, 1
    c, h, w = x.shape[1:]
    n_features = c * h * w
    n_ex_total = x.shape[0]
    x, y = x[corr_classified], y[corr_classified]

    ### initialization
    delta_init = np.zeros(x.shape)
    s = h // 5
    log.print('Initial square side={} for bumps'.format(s))
    sp_init = (h - s * 5) // 2
    center_h = sp_init + 0
    for counter in range(h // s):
        center_w = sp_init + 0
        for counter2 in range(w // s):
            delta_init[:, :, center_h:center_h + s, center_w:center_w + s] += meta_pseudo_gaussian_pert(s).reshape(
                [1, 1, s, s]) * np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])
            center_w += s
        center_h += s

    x_best = np.clip(x + delta_init / np.sqrt(np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * eps, 0, 1)

    logits = model.predict(preprocess(x_best))
    loss_min = model.loss(y, logits, targeted, loss_type=loss_type)
    margin_min = model.loss(y, logits, targeted, loss_type='margin_loss')
    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

    time_start = time.time()
    s_init = int(np.sqrt(p_init * n_features / c))
    metrics = np.zeros([n_iters, 7])
    for i_iter in range(n_iters):
        idx_to_fool = (margin_min > 0.0)

        x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
        y_curr, margin_min_curr = y[idx_to_fool], margin_min[idx_to_fool]
        loss_min_curr = loss_min[idx_to_fool]
        delta_curr = x_best_curr - x_curr

        p = p_selection(p_init, i_iter, n_iters)
        s = max(int(round(np.sqrt(p * n_features / c))), 3)

        if s % 2 == 0:
            s += 1

        s2 = s + 0
        ### window_1
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        new_deltas_mask = np.zeros(x_curr.shape)
        new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0

        ### window_2
        center_h_2 = np.random.randint(0, h - s2)
        center_w_2 = np.random.randint(0, w - s2)
        new_deltas_mask_2 = np.zeros(x_curr.shape)
        new_deltas_mask_2[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0
        norms_window_2 = np.sqrt(
            np.sum(delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] ** 2, axis=(-2, -1),
                   keepdims=True))

        ### compute total norm available
        curr_norms_window = np.sqrt(
            np.sum(((x_best_curr - x_curr) * new_deltas_mask) ** 2, axis=(2, 3), keepdims=True))
        curr_norms_image = np.sqrt(np.sum((x_best_curr - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))
        mask_2 = np.maximum(new_deltas_mask, new_deltas_mask_2)
        norms_windows = np.sqrt(np.sum((delta_curr * mask_2) ** 2, axis=(2, 3), keepdims=True))

        ### create the updates
        new_deltas = np.ones([x_curr.shape[0], c, s, s])
        new_deltas = new_deltas * meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
        new_deltas *= np.random.choice([-1, 1], size=[x_curr.shape[0], c, 1, 1])
        old_deltas = delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window)
        new_deltas += old_deltas
        new_deltas = new_deltas / np.sqrt(np.sum(new_deltas ** 2, axis=(2, 3), keepdims=True)) * (
            np.maximum(eps ** 2 - curr_norms_image ** 2, 0) / c + norms_windows ** 2) ** 0.5
        delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 0.0  # set window_2 to 0
        delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] = new_deltas + 0  # update window_1

        hps_str = 's={}->{}'.format(s_init, s)
        x_new = x_curr + delta_curr / np.sqrt(np.sum(delta_curr ** 2, axis=(1, 2, 3), keepdims=True)) * eps
        x_new = np.clip(x_new, min_val, max_val)
        curr_norms_image = np.sqrt(np.sum((x_new - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))

        logits = model.predict(x_new)
        loss = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = model.loss(y_curr, logits, targeted, loss_type='margin_loss')

        idx_improved = loss < loss_min_curr
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr

        idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        acc = (margin_min > 0.0).sum() / n_ex_total
        acc_corr = (margin_min > 0.0).mean()
        mean_nq, mean_nq_ae, median_nq, median_nq_ae = np.mean(n_queries), np.mean(
            n_queries[margin_min <= 0]), np.median(n_queries), np.median(n_queries[margin_min <= 0])

        time_total = time.time() - time_start
        log.print(
            '{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.1f} med#q_ae={:.1f} {}, n_ex={}, {:.0f}s, loss={:.3f}, max_pert={:.1f}, impr={:.0f}'.
                format(i_iter + 1, acc, acc_corr, mean_nq_ae, median_nq_ae, hps_str, x.shape[0], time_total,
                       np.mean(margin_min), np.amax(curr_norms_image), np.sum(idx_improved)))
        metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq, margin_min.mean(), time_total]
        if (i_iter <= 500 and i_iter % 500) or (i_iter > 100 and i_iter % 500) or i_iter + 1 == n_iters or acc == 0:
            np.save(metrics_path, metrics)
        if acc == 0:
            curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
            print('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))
            break

    curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
    print('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))

    return n_queries, x_best


def square_attack_linf(model, x, y, corr_classified, eps, n_iters, p_init, metrics_path, targeted, loss_type):
    """ The Linf square attack """
    np.random.seed(0)  # important to leave it here as well
    min_val, max_val = 0, 1 # if x.max() <= 1 else 255

    n_ex_total = len(x) # x.shape[0]

    for i, cor in enumerate(corr_classified):
        if cor is False:
            x.pop(i)
            y.pop(i)
    # x, y = np.array(x)[corr_classified], np.array(y)[corr_classified]

    # [c, 1, w], i.e. vertical stripes work best for untargeted attacks

    x_best = []
    for i, xi in enumerate(x):
        c, h, w = xi.shape[:]
        init_delta = np.random.choice([-eps, eps], size=[len(x), c, 1, w])
        x_best.append(np.clip(xi + init_delta[i], min_val, max_val))
    # x_best = np.clip(x + init_delta, min_val, max_val)

    logits = []
    for xb in x_best:
        logits.append(model.predict(preprocess(torch.tensor(xb)).numpy())[0])
    logits = np.array(logits)

    loss_min = model.loss(y, logits, targeted, loss_type=loss_type)
    margin_min = model.loss(y, logits, targeted, loss_type='margin_loss')
    n_queries = np.ones(len(x))  # ones because we have already used 1 query

    time_start = time.time()
    metrics = np.zeros([n_iters, 7])
    for i_iter in range(n_iters - 1):
        idx_to_fool = margin_min > 0
        idx_to_fool = [i for i, x in enumerate(idx_to_fool) if x]

        x_curr = []
        x_best_curr = []
        y_curr = []
        for idx in idx_to_fool:
            x_curr.append(x[idx])
            x_best_curr.append(x_best[idx])
            y_curr.append(y[idx])

        # x_curr, x_best_curr, y_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool]

        loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
        deltas = [ (xb - xc) for xb, xc in zip(x_best_curr, x_curr) ]

        p = p_selection(p_init, i_iter, n_iters)
        for i_img in range(len(x_best_curr)):
            c, h, w = x[i_img].shape[:]
            n_features = c*h*w

            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img][:, center_h:center_h+s, center_w:center_w+s]
            x_best_curr_window = x_best_curr[i_img][:, center_h:center_h+s, center_w:center_w+s]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img][:, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                deltas[i_img][:, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-eps, eps], size=[c, 1, 1])

        x_new = [  np.clip(xc + d, min_val, max_val) for xc, d in zip(x_curr, deltas) ]
        # x_new = np.clip(x_curr + deltas, min_val, max_val)

        logits = []
        for xn in x_new:
            logits.append(model.predict(preprocess(torch.tensor(xn)).numpy())[0])
        logits = np.array(logits)
    
        # logits = model.predict(preprocess(torch.tensor(x_new)).numpy())
    
        loss = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = model.loss(y_curr, logits, targeted, loss_type='margin_loss')

        idx_improved = loss < loss_min_curr
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr

        idx_improved = np.reshape(idx_improved, [-1, *[1]*3])
        # idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])

        for i in range(len(idx_improved)):
            x_best[idx_to_fool[i]] = idx_improved[i] * x_new[i] + ~idx_improved[i] * x_best_curr[i]
        # x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr

        n_queries[idx_to_fool] += 1

        if len(margin_min) > 0 and len(n_queries) > 0:
            acc = (margin_min > 0.0).sum() / n_ex_total
            acc_corr = (margin_min > 0.0).mean()
            mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0]), np.median(n_queries[margin_min <= 0])
            avg_margin_min = np.mean(margin_min)
            time_total = time.time() - time_start
            log.print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.2f} med#q={:.1f}, avg_margin={:.2f} (n_ex={}, eps={:.3f}, {:.2f}s)'.
                format(i_iter+1, acc, acc_corr, mean_nq_ae, median_nq_ae, avg_margin_min, len(x), eps, time_total))

            metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq_ae, margin_min.mean(), time_total]

        if (i_iter <= 500 and i_iter % 20 == 0) or (i_iter > 100 and i_iter % 50 == 0) or i_iter + 1 == n_iters or acc == 0:
            np.save(metrics_path, metrics)
        if acc == 0:
            break

    return n_queries, x_best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--model', type=str, default='pt_resnet', choices=models.all_model_names, help='Model name.')
    parser.add_argument('--attack', type=str, default='square_linf', choices=['square_linf', 'square_l2'], help='Attack.')
    parser.add_argument('--exp_folder', type=str, default='exps', help='Experiment folder to store all output.')
    parser.add_argument('--gpu', type=str, default='7', help='GPU number. Multiple GPUs are possible for PT models.')
    parser.add_argument('--n_ex', type=int, default=10000, help='Number of test ex to test on.')
    parser.add_argument('--p', type=float, default=0.05,
                        help='Probability of changing a coordinate. Note: check the paper for the best values. '
                             'Linf standard: 0.05, L2 standard: 0.1. But robust models require higher p.')
    parser.add_argument('--eps', type=float, default=0.05, help='Radius of the Lp ball.')
    parser.add_argument('--n_iter', type=int, default=10000)
    parser.add_argument('--targeted', action='store_true', help='Targeted or untargeted attack.')
    args = parser.parse_args()
    args.loss = 'margin_loss' if not args.targeted else 'cross_entropy'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dataset = 'mnist' if 'mnist' in args.model else 'cifar10' if 'cifar10' in args.model else 'imagenet'
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    hps_str = '{}_{}_{}_{}_n_ex_eps_p_n_iter_'.format(
        timestamp, args.model, dataset, args.attack, args.n_ex, args.eps, args.p, args.n_iter)
    args.eps = args.eps / 255.0 if dataset == 'imagenet' else args.eps  # for mnist and cifar10 we leave as it is
    batch_size = data.bs_dict[dataset]
    model_type = 'deepapi' if 'deepapi' in args.model else 'pt' if 'pt_' in args.model else 'tf'
    n_cls = 1000 if dataset == 'imagenet' else 10
    gpu_memory = 0.5 if dataset == 'mnist' and args.n_ex > 1000 else 0.15 if dataset == 'mnist' else 0.99

    log_path = '{}/{}.log'.format(args.exp_folder, hps_str)
    metrics_path = '{}/{}.metrics'.format(args.exp_folder, hps_str)

    log = utils.Logger(log_path)
    log.print('All hps: {}'.format(hps_str))

    if args.model != 'pt_inception':
        x_test, y_test = data.datasets_dict[dataset](args.n_ex)
    else:  # exception for inception net on imagenet -- 299x299 images instead of 224x224
        x_test, y_test = data.datasets_dict[dataset](args.n_ex, size=299)
    x_test, y_test = x_test[:args.n_ex], y_test[:args.n_ex]

    if args.model == 'pt_post_avg_cifar10':
        x_test /= 255.0
        args.eps = args.eps / 255.0

    models_class_dict = {'tf': models.ModelTF, 'pt': models.ModelPT, 'deepapi_vgg16': models.VGG16ImageNet}

    if model_type == 'deepapi':
        model = models_class_dict[args.model]('http://localhost:8080/vgg16')
    else:
        model = models_class_dict[model_type](args.model, batch_size, gpu_memory)

    if model_type == 'deepapi':
        preprocess = nchw_to_nhwc
    else:
        preprocess = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                ])

    logits_clean = []
    corr_classified = []
    for i, x in enumerate(x_test):
        logits = model.predict(preprocess(torch.tensor(x)).numpy())

        logits_clean.append(logits)
        corr_classified.append((logits.argmax(1) == y_test[i])[0])
        if logits.argmax(1) != y_test[i]:
            # im = Image.fromarray(np.array(np.uint8(preprocess(torch.tensor(x)).numpy()[0]*255.0)))
            # im.show()
            print('Incorrectly classified: {} {} as {}'.format(i, model.imagenet_labels[y_test[i]], model.imagenet_labels[logits.argmax(1)[0]]))
    logits_clean = np.array(logits_clean)

    # important to check that the model was restored correctly and the clean accuracy is high
    log.print('Clean accuracy: {:.2%}'.format(np.mean(corr_classified)))

    square_attack = square_attack_linf if args.attack == 'square_linf' else square_attack_l2
    y_target = utils.random_classes_except_current(y_test, n_cls) if args.targeted else y_test
    y_target_onehot = utils.dense_to_onehot(y_target, n_cls=n_cls)
    # Note: we count the queries only across correctly classified images
    n_queries, x_adv = square_attack(model, x_test, y_target_onehot, corr_classified, args.eps, args.n_iter,
                                     args.p, metrics_path, args.targeted, args.loss)

    for i, x in enumerate(x_adv):
        im = Image.fromarray(np.array(np.uint8(preprocess(torch.tensor(x_test[i])).numpy()[0]*255.0)))
        im_adv = Image.fromarray(np.array(np.uint8(preprocess(torch.tensor(x)).numpy()[0]*255.0)))
        im.save(f"images/x_{i}.jpg")
        im_adv.save(f"images/x_{i}_adv.jpg")
