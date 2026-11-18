# Patched copy of MedMNIST2D/train_and_eval_pytorch.py from https://github.com/MedMNIST/experiments
# (commit 70b6b3a7ad7afddff1df2a3b735235830fbdb142, Apache License 2.0).
#
# Unchanged from the released script: model (torchvision ResNet-18/50, no pretraining), data
# transform (28 -> 224 with PIL NEAREST, ToTensor, Normalize(0.5, 0.5)), loss, Adam lr 0.001,
# MultiStepLR at 50% and 75% of the epochs (gamma 0.1), epochs, batch size, model selection on
# the best validation AUC, and the final evaluation with medmnist.Evaluator.
#
# Every change is marked "PATCH". The changes only affect where and how the run executes:
#   * checkpoint + resume, so a run can span several 12-hour Kaggle/Colab sessions
#   * --deadline: stop cleanly (exit code 3) before a session time limit
#   * --gpu_ids 1 now uses the second GPU (the released script then asks for a GPU it has hidden);
#     --gpu_ids mps runs on an Apple-silicon Mac, --gpu_ids cpu on the CPU
#   * fixed output folder <output_root>/<data_flag>/<run>/ instead of a timestamp, plus result.json
#   * optional --seed, --workers (parallel data loading), --no_weights, --save_last (also keep the
#     last-epoch weights as last_model.pth); history.json with every epoch's numbers
#   * optional --amp: float16 mixed precision on CUDA (changes the arithmetic slightly)
#   * with --amp, two guards for a float16 overflow (added after one PathMNIST run crashed on NaN
#     validation scores): a training batch whose loss is not finite gets its BatchNorm running
#     statistics restored (GradScaler already skips its weight update), and an evaluation that
#     gives non-finite scores in float16 is redone in float32. Neither does anything when no
#     overflow happens, so runs without one train exactly as before.
#   * optional --skip_logging_evals: skip the per-epoch train-set and test-set evaluations, which
#     the released script only writes to TensorBoard. The validation set is still evaluated every
#     epoch for model selection, and the final train/val/test evaluation of the best model is
#     unchanged, so the reported numbers are computed exactly as before.

import argparse
import json  # PATCH
import math  # PATCH
import os
import platform  # PATCH
import random  # PATCH
import sys  # PATCH
import time
from collections import OrderedDict
from copy import deepcopy

import medmnist
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from medmnist import INFO, Evaluator
from models import ResNet18, ResNet50
from tensorboardX import SummaryWriter
from torchvision.models import resnet18, resnet50
from tqdm import trange

EXIT_PAUSED = 3  # PATCH: exit code when the run stopped before --deadline and can be resumed


# ---------------------------------------------------------------- PATCH: helpers
def pick_device(gpu_ids):
    """'0' or '1' -> that CUDA GPU, 'mps' -> Apple GPU, 'cpu' or '-1' -> CPU.
    The released script set CUDA_VISIBLE_DEVICES to the chosen id and then asked for cuda:<id>,
    which fails for any id other than 0."""
    s = str(gpu_ids).strip().lower()
    if s == 'mps':
        return torch.device('mps')
    if s in ('cpu', '', '-1'):
        return torch.device('cpu')
    first = int(s.split(',')[0])
    return torch.device('cpu') if first < 0 else torch.device('cuda:{}'.format(first))


def get_rng_state(device):
    state = {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate()}
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state, device):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if device.type == 'cuda' and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def consume_loader_rng(loader):
    """Starting a pass over a DataLoader draws one number from the global torch RNG
    (_BaseDataLoaderIter.__init__ draws its base seed). Skipping an evaluation pass would shift
    the shuffling of every later epoch, so draw the same number here. With this, a skipped
    evaluation leaves training identical to the released script."""
    torch.empty((), dtype=torch.int64).random_(generator=loader.generator)


def make_grad_scaler():
    try:
        return torch.amp.GradScaler('cuda')
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def atomic_torch_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, download, model_flag, resize, as_rgb, model_path, run,
         seed=None, amp=False, skip_logging_evals=False, workers=0, deadline=None, no_weights=False, ckpt_every=600,
         save_last=False):  # PATCH: new options

    lr = 0.001
    gamma=0.1
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]

    info = INFO[data_flag]
    task = info['task']
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    # PATCH: device selection (see pick_device); the released script overwrote CUDA_VISIBLE_DEVICES
    device = pick_device(gpu_ids)
    if device.type == 'cuda':
        torch.cuda.set_device(device)  # PATCH: pinned-memory thread and default CUDA ops use this GPU too
    if amp and device.type != 'cuda':
        print('[patch] --amp is only used on CUDA GPUs; running in float32 on', device)
        amp = False
    if amp:
        torch.backends.cudnn.benchmark = True
    channels_last = amp

    # PATCH: fixed run folder so an interrupted run can be found and resumed
    output_root = os.path.join(output_root, data_flag, run)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    result_path = os.path.join(output_root, 'result.json')
    ckpt_path = os.path.join(output_root, 'checkpoint.pt')
    progress_path = os.path.join(output_root, 'progress.json')
    history_path = os.path.join(output_root, 'history.json')
    if model_path is None and num_epochs > 0 and os.path.exists(result_path):
        print('[patch] already finished:', result_path)
        return 0

    # PATCH: optional seed (the released script sets none)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    print('==> Preparing data...')

    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    else:
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])

    train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb, size=size)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb, size=size)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb, size=size)

    # PATCH: optional worker processes and pinned memory; batches and their order are unchanged
    loader_kw = dict(num_workers=workers, pin_memory=(device.type == 'cuda'))
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True, **loader_kw)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=False, **loader_kw)
    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False, **loader_kw)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False, **loader_kw)

    print('==> Building and training model...')


    if model_flag == 'resnet18':
        model =  resnet18(pretrained=False, num_classes=n_classes) if resize else ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'resnet50':
        model =  resnet50(pretrained=False, num_classes=n_classes) if resize else ResNet50(in_channels=n_channels, num_classes=n_classes)
    else:
        raise NotImplementedError

    model = model.to(device)
    if channels_last:  # PATCH (AMP only)
        model = model.to(memory_format=torch.channels_last)

    train_evaluator = medmnist.Evaluator(data_flag, 'train', size=size)
    val_evaluator = medmnist.Evaluator(data_flag, 'val', size=size)
    test_evaluator = medmnist.Evaluator(data_flag, 'test', size=size)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    evkw = dict(amp=amp, channels_last=channels_last)  # PATCH

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)
        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root, **evkw)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run, output_root, **evkw)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run, output_root, **evkw)

        print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
              'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
              'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))

    if num_epochs == 0:
        return 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    scaler = make_grad_scaler() if amp else None  # PATCH

    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)

    writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

    best_auc = 0
    best_epoch = 0
    best_model = deepcopy(model)

    global iteration
    iteration = 0

    # ------------------------------------------------------------ PATCH: resume
    settings = dict(data_flag=data_flag, model_flag=model_flag, resize=resize, as_rgb=as_rgb, size=size,
                    batch_size=batch_size, num_epochs=num_epochs, amp=amp, seed=seed)
    start_epoch = 0
    history = []
    sessions = 1
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except TypeError:  # PyTorch older than 1.13 has no weights_only argument
            ckpt = torch.load(ckpt_path, map_location='cpu')
        if ckpt['settings'] != settings:
            sys.exit('[patch] checkpoint settings %s do not match this command %s; '
                     'use another --output_root or delete %s' % (ckpt['settings'], settings, ckpt_path))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if scaler is not None:
            scaler.load_state_dict(ckpt['scaler'])
        best_model.load_state_dict(ckpt['best_model'])
        best_auc = ckpt['best_auc']
        best_epoch = ckpt['best_epoch']
        iteration = ckpt['iteration']
        history = ckpt['history']
        sessions = ckpt.get('sessions', 1) + 1
        start_epoch = ckpt['epoch'] + 1
        set_rng_state(ckpt['rng'], device)
        print('[patch] resumed from epoch %d of %d (best val AUC so far %.5f at epoch %d)'
              % (start_epoch, num_epochs, best_auc, best_epoch))

    def save_checkpoint(epoch):
        atomic_torch_save({
            'settings': settings, 'epoch': epoch,
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict() if scaler is not None else None,
            'best_model': best_model.state_dict(), 'best_auc': best_auc, 'best_epoch': best_epoch,
            'iteration': iteration, 'history': history, 'sessions': sessions,
            'rng': get_rng_state(device)}, ckpt_path)
        secs = [h['seconds'] for h in history]
        atomic_json({'epochs_done': epoch + 1, 'num_epochs': num_epochs,
                     'mean_epoch_seconds': float(np.mean(secs[-5:])) if secs else None,
                     'eval_seconds_per_sample': history[-1]['eval_seconds_per_sample'] if history else None,
                     'best_val_auc': best_auc, 'best_epoch': best_epoch, 'updated': time.time()}, progress_path)
        atomic_json(history, history_path)

    n_eval_final = len(train_dataset) + len(val_dataset) + len(test_dataset)
    last_saved_epoch = start_epoch - 1
    last_save_time = time.time()
    # ------------------------------------------------------------ end PATCH

    for epoch in trange(start_epoch, num_epochs):  # PATCH: start at start_epoch

        # PATCH: stop before the deadline if the next epoch (and the final evaluation) might not fit
        if deadline is not None and history:
            recent = [h['seconds'] for h in history[-3:]]
            need = max(recent) * 1.15 + 30
            if epoch == num_epochs - 1:
                need += history[-1]['eval_seconds_per_sample'] * n_eval_final * 1.3 + 60
            if time.time() + need > deadline:
                if last_saved_epoch != epoch - 1:
                    save_checkpoint(epoch - 1)
                writer.close()
                print('[patch] PAUSED before epoch %d to stay inside the deadline; rerun the same command to resume.' % epoch)
                return EXIT_PAUSED
        t_epoch = time.time()

        train_loss = train(model, train_loader, task, criterion, optimizer, device, writer, scaler=scaler, channels_last=channels_last)

        if skip_logging_evals:  # PATCH
            consume_loader_rng(train_loader_at_eval)
            train_metrics = [float('nan')] * 3
        else:
            train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, **evkw)
        t_val = time.time()
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run, **evkw)
        eval_seconds_per_sample = (time.time() - t_val) / len(val_dataset)  # PATCH
        if skip_logging_evals:  # PATCH
            consume_loader_rng(test_loader)
            test_metrics = [float('nan')] * 3
        else:
            test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run, **evkw)

        scheduler.step()

        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        for key, value in log_dict.items():
            if not (isinstance(value, float) and np.isnan(value)):  # PATCH: skipped evaluations are not logged
                writer.add_scalar(key, value, epoch)

        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)
            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)

        # PATCH: per-epoch record, progress line and checkpoint
        seconds = time.time() - t_epoch
        history.append({'epoch': epoch, 'train_loss': float(train_loss), 'val_auc': float(val_metrics[1]),
                        'val_acc': float(val_metrics[2]), 'test_auc': float(test_metrics[1]),
                        'test_acc': float(test_metrics[2]), 'lr': optimizer.param_groups[0]['lr'],
                        'seconds': seconds, 'eval_seconds_per_sample': eval_seconds_per_sample})
        mem = ''
        if device.type == 'cuda':
            mem = '  peak GPU memory %.1f GiB' % (torch.cuda.max_memory_allocated(device) / 2**30)
        line = '[epoch %d/%d] train_loss %.4f  val_auc %.5f  val_acc %.5f  %.0fs (%.0f train img/s)%s' % (
            epoch + 1, num_epochs, train_loss, val_metrics[1], val_metrics[2], seconds,
            len(train_dataset) / max(seconds, 1e-9), mem)
        print(line, flush=True)
        with open(os.path.join(output_root, 'epochs.log'), 'a') as f:
            f.write(line + '\n')
        if epoch == num_epochs - 1 or time.time() - last_save_time >= ckpt_every:
            save_checkpoint(epoch)
            last_saved_epoch = epoch
            last_save_time = time.time()

    state = {
        'net': best_model.state_dict(),
    }

    path = os.path.join(output_root, 'best_model.pth')
    if not no_weights:  # PATCH: optional
        torch.save(state, path)
        if save_last:  # PATCH: optional last-epoch weights, same format
            torch.save({'net': model.state_dict()}, os.path.join(output_root, 'last_model.pth'))

    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root, **evkw)
    val_metrics = test(best_model, val_evaluator, val_loader, task, criterion, device, run, output_root, **evkw)
    test_metrics = test(best_model, test_evaluator, test_loader, task, criterion, device, run, output_root, **evkw)

    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

    log = '%s\n' % (data_flag) + train_log + val_log + test_log
    print(log)

    with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
        f.write(log)

    writer.close()

    # PATCH: machine-readable result for collect_results.py, then drop the large checkpoint
    atomic_json(history, history_path)
    secs = [h['seconds'] for h in history]
    atomic_json({
        'data_flag': data_flag, 'run': run, 'seed': seed, 'model_flag': model_flag, 'resize': resize,
        'as_rgb': as_rgb, 'size': size, 'batch_size': batch_size, 'num_epochs': num_epochs, 'lr': lr,
        'milestones': milestones, 'amp': amp, 'skip_logging_evals': skip_logging_evals, 'workers': workers,
        'best_epoch': best_epoch, 'best_val_auc': float(best_auc),
        'train_auc': float(train_metrics[1]), 'train_acc': float(train_metrics[2]),
        'val_auc': float(val_metrics[1]), 'val_acc': float(val_metrics[2]),
        'test_auc': float(test_metrics[1]), 'test_acc': float(test_metrics[2]),
        'train_hours': float(np.sum(secs)) / 3600, 'sessions': sessions,
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else device.type,
        'torch': torch.__version__, 'torchvision': torchvision.__version__, 'medmnist': medmnist.__version__,
        'python': platform.python_version(), 'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'script': os.path.basename(__file__)}, result_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return 0


def train(model, train_loader, task, criterion, optimizer, device, writer, scaler=None, channels_last=False):
    total_loss = []
    global iteration

    # PATCH (AMP only): a float16 overflow makes a batch's loss non-finite. GradScaler then skips that
    # weight update, but the forward pass has already written the bad values into the BatchNorm running
    # statistics, which turns every later evaluation into NaN. Keep a copy and put it back for such a batch.
    bn_layers = [m for m in model.modules()
                 if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.running_mean is not None] if scaler is not None else []
    overflow_batches = 0

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs.to(device, non_blocking=True)  # PATCH: non_blocking (pinned memory)
        if channels_last:  # PATCH (AMP only)
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        if bn_layers:  # PATCH
            bn_saved = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone()) for m in bn_layers]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):  # PATCH
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)

        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:  # PATCH: mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # PATCH: overflow; restore the BatchNorm statistics (after backward, which still needs them as they were)
        if bn_layers and not math.isfinite(total_loss[-1]):
            with torch.no_grad():
                for m, (rm, rv, nbt) in zip(bn_layers, bn_saved):
                    m.running_mean.copy_(rm)
                    m.running_var.copy_(rv)
                    m.num_batches_tracked.copy_(nbt)
            overflow_batches += 1

    if overflow_batches:  # PATCH: the epoch's loss is then the mean over the batches with a finite loss
        print('[patch] float16 overflow in %d of %d batches: their weight updates were skipped and the '
              'BatchNorm statistics restored' % (overflow_batches, len(total_loss)), flush=True)
        total_loss = [x for x in total_loss if math.isfinite(x)] or [float('nan')]
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss


def test(model, evaluator, data_loader, task, criterion, device, run, save_folder=None, amp=False, channels_last=False):

    model.eval()

    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device, non_blocking=True)  # PATCH
            if channels_last:  # PATCH (AMP only)
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):  # PATCH
                outputs = model(inputs)
            outputs = outputs.float()  # PATCH: scores in float32 (no-op without AMP)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
        if amp and not np.isfinite(y_score).all():  # PATCH: float16 overflow in this evaluation; redo it in float32
            print('[patch] non-finite scores in float16; evaluating this split again in float32', flush=True)
            return test(model, evaluator, data_loader, task, criterion, device, run, save_folder,
                        amp=False, channels_last=channels_last)
        auc, acc = evaluator.evaluate(y_score, save_folder, run)

        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST2D')

    parser.add_argument('--data_flag',
                        default='pathmnist',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=100,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--size',
                        default=28,
                        help='the image size of the dataset, 28 or 64 or 128 or 224, default=28',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        help='PATCH: 0, 1, ... for a CUDA GPU, mps for an Apple-silicon Mac, cpu for the CPU',
                        type=str)
    parser.add_argument('--batch_size',
                        default=128,
                        type=int)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--resize',
                        help='resize images of size 28x28 to 224x224',
                        action="store_true")
    parser.add_argument('--as_rgb',
                        help='convert the grayscale image to RGB',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default='resnet18',
                        help='choose backbone from resnet18, resnet50',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)
    # PATCH: new options (all off by default, so the default behaviour matches the released script)
    parser.add_argument('--seed', default=None, type=int, help='PATCH: seed python, numpy and torch')
    parser.add_argument('--amp', action='store_true', help='PATCH: float16 mixed precision on CUDA')
    parser.add_argument('--skip_logging_evals', action='store_true',
                        help='PATCH: skip the per-epoch train/test evaluations that are only logged to TensorBoard')
    parser.add_argument('--workers', default=0, type=int, help='PATCH: DataLoader worker processes')
    parser.add_argument('--deadline', default=None, type=float,
                        help='PATCH: unix time; stop cleanly (exit code 3) before it, rerun to resume')
    parser.add_argument('--no_weights', action='store_true', help='PATCH: do not write best_model.pth')
    parser.add_argument('--ckpt_every', default=600, type=float,
                        help='PATCH: seconds between checkpoints (one is always written at the end and before a pause)')
    parser.add_argument('--save_last', action='store_true',
                        help='PATCH: also save the last-epoch weights as last_model.pth')


    args = parser.parse_args()
    data_flag = args.data_flag
    output_root = args.output_root
    num_epochs = args.num_epochs
    size = args.size
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    download = args.download
    model_flag = args.model_flag
    resize = args.resize
    as_rgb = args.as_rgb
    model_path = args.model_path
    run = args.run

    code = main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, download, model_flag, resize, as_rgb, model_path, run,
                seed=args.seed, amp=args.amp, skip_logging_evals=args.skip_logging_evals, workers=args.workers,
                deadline=args.deadline, no_weights=args.no_weights, ckpt_every=args.ckpt_every,
                save_last=args.save_last)  # PATCH
    sys.exit(code or 0)  # PATCH
