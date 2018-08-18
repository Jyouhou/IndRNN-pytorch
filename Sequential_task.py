import os
import pdb
import shutil
import argparse
import torch
import torch.nn as nn
import torch.functional as f
import torch.nn.functional as F
import torchvision.transforms as transform
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import indRNN
import numpy as np
import math
import time
from collections import OrderedDict
import shutil
import glob


def backup(files, folder):
    for file in files:
        shutil.copy(file, os.path.join(folder, file.split("/")[-1]))
    print(f"Script backup in '{folder}'")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--lr_setting', default="2e-4/600000/0.1", type=str,
                        help='Learning rate and decaying scheme: lr_rate/decay_step/decay_rate')
    parser.add_argument('-m', '--momentum', default=0.9, type=float, help="Momentum for batch norm")
    parser.add_argument('-s', '--seed', default=123, type=int, help='Random seed')
    parser.add_argument('--hidden_size', default=128, type=int, help='Hidden embedding size')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Mini-batch size for training')
    parser.add_argument('--epochs', default=2000, type=int, help='Number of epochs')
    parser.add_argument('--dataset', default='mnist', type=str,
                        help='Model name')
    parser.add_argument('--log_step', default=1, type=int,
                        help='step')
    parser.add_argument('--inpd', default=1, type=int,
                        help='INPUT_DIMENSION')
    parser.add_argument('-t', '--temperature', default=10, type=int,
                        help='temperature')
    parser.add_argument('--num_layer', default=6, type=int,
                        help='step')
    parser.add_argument('--log', default="exp.log", type=str,
                        help='log file')
    parser.add_argument('--save', default="model.pt", type=str,
                        help='')
    parser.add_argument('--optim', default="adam", type=str,
                        help='')
    parser.add_argument('--init_ih', default="uniform/-0.001/0.001", type=str,
                        help='1. constant/value ; 2. uniform/lower/upper; 3. norm/mean/std. Example: constant/1.0, uniform/-0.001/0.001, norm/0.0/0.001')
    parser.add_argument('--log_folder', default="./experiment/", type=str,
                        help='')
    parser.add_argument('--bid', action="store_true")
    parser.add_argument('--bidadd', action="store_true")
    parser.add_argument('--mutadd', action="store_true")
    parser.add_argument('--last_hidden', action="store_false")
    parser.add_argument('--bn', action="store_true")
    parser.add_argument('--ln', action="store_true")
    parser.add_argument('--min_grad', default=-5, type=float)
    parser.add_argument('--max_grad', default=5, type=float)
    parser.add_argument('--g_clip', action="store_false")
    parser.add_argument('--bias', action="store_false")
    parser.add_argument('--mul_label', action="store_true")
    parser.add_argument('--act', default="relu", type=str,
                        help='')
    parser.add_argument('--debug', action="store_true", help="Set to True to use pdb debugger")
    parser.add_argument('--dropout', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--test_model', default="", type=str,
                        help='')
    parser.add_argument('--permutator', default="", type=str,
                        help='')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='l2 penalty')
    args = parser.parse_args()
    assert (not (args.bn and args.ln))
    if not(args.bn or args.ln):
        args.bn = True
    args.learning_rate, args.decay_step, args.decay_rate = args.lr_setting.split("/")
    args.learning_rate = float(args.learning_rate)
    args.decay_step = int(args.decay_step)
    args.decay_rate = float(args.decay_rate)
    attr_file = os.path.join(args.log_folder, "args.txt")
    if not os.path.isdir(args.log_folder):
        os.mkdir(args.log_folder)
    with open(attr_file, "w")as f:
        for name in dir(args):
            if name[0] != "_":
                f.write(f"--{name}={getattr(args, name)} ")
    print(f"args saved to: '{attr_file}'")
    if args.dataset == "cifar":
        args.inpd = 3
    return args


args = parse()
torch.manual_seed(args.seed)
CUDA = torch.cuda.is_available()
LOG = open(os.path.join(args.log_folder, args.log), "w") if len(args.log) > 0 else None
OSTREAM = {LOG, None}
STEP_COUNT = 0

backup(glob.glob("./*.py"), args.log_folder)

TIME_STEP = 1024 if args.dataset == "cifar" else 784
hidden_min_abs = (1 / 2)**(1 / TIME_STEP)  # ?
hidden_max_abs = 2**(1 / TIME_STEP)


class DataLoader(object):
    def __init__(self, task="mnist", transform_pixel="default"):
        dataset = {"mnist": MNIST,
                   "pmnist": MNIST,
                   "FashionMNIST": FashionMNIST,
                   "cifar": CIFAR10}
        dataset_name = {
            "mnist": "mnist",
            "pmnist": "mnist",
            "FashionMNIST": "FashionMNIST",
            "cifar": "CIFAR10"
        }

        transform_scheme = transform.Compose([
            transform.ToTensor(),
            transform.Lambda(lambda x:2 * (x / 1.) - 1) if transform_pixel == "default" else transform.Normalize((0.1307,), (0.3081,)),
        ])

        # training set
        self.train_set = dataset[task](
            root=f'./data{dataset_name[task]}',
            train=True,
            download=True,
            transform=transform_scheme)
        training_set_size = self.train_set.train_data.shape[0]
        if task == "cifar":
            self.train_set.train_data = np.reshape(self.train_set.train_data, (training_set_size, -1, 3))
        else:
            self.train_set.train_data = self.train_set.train_data.view(self.train_set.train_data.size(0), -1, 1)  # dataset_size x 784 x 1
        self.train_set.train_data = self.train_set.train_data[:(training_set_size * 4 // 5)]
        self.train_set.train_labels = self.train_set.train_labels[:(training_set_size * 4 // 5)]

        # validation set
        self.valid_set = dataset[task](
            root=f'./data{dataset_name[task]}',
            train=True,
            download=True,
            transform=transform_scheme)
        if task == "cifar":
            self.valid_set.train_data = np.reshape(self.valid_set.train_data, (training_set_size, -1, 3))
        else:
            self.valid_set.train_data = self.valid_set.train_data.view(self.valid_set.train_data.size(0), -1, 1)
        self.valid_set.train_data = self.valid_set.train_data[(training_set_size * 4 // 5):]
        self.valid_set.train_labels = self.valid_set.train_labels[(training_set_size * 4 // 5):]

        # test set
        self.test_set = dataset[task](
            root=f'./data{dataset_name[task]}',
            train=False,
            download=True,
            transform=transform_scheme)
        if task == "cifar":
            self.test_set.test_data = np.reshape(self.test_set.test_data, (self.test_set.test_data.shape[0], -1, 3))
        else:
            self.test_set.test_data = self.test_set.test_data.view(self.test_set.test_data.size(0), -1, 1)

        if task == "pmnist":
            if args.test:
                self.Perm = torch.load(os.path.join(args.permutator))
            else:
                self.Perm = torch.randperm(784)
                torch.save(self.Perm, os.path.join(args.log_folder, "permutator.pt"))
            self.train_set.train_data = self.train_set.train_data[:, self.Perm, :]
            self.valid_set.train_data = self.valid_set.train_data[:, self.Perm, :]
            self.test_set.test_data = self.test_set.test_data[:, self.Perm, :]

    def generator(self, Set='train', batch_size=32):
        def _generator(data_iter):
            for image, label in data_iter:
                _ = image.squeeze_(1)
                image = image.contiguous().view(image.size(0), TIME_STEP, -1)  # batch * time step * channel
                if CUDA:
                    image = image.cuda()
                    label = label.cuda()
                yield image, label
        if Set in ['train', 'valid', 'test']:
            return _generator(torch.utils.data.DataLoader(
                {'train': self.train_set, 'valid': self.valid_set,
                    'test': self.test_set}[Set],
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=10
            ))
        else:
            raise ValueError('select "train", "valid", or "test"')


def train(model, data, optimizer, lr_scheduler):
    model.train()
    acc_loss = []
    acc_acc = []
    ma_loss = None
    ma_acc = None
    global STEP_COUNT
    t = time.time()
    for Iter, (image, label) in enumerate(data):
        lr_scheduler.step(epoch=STEP_COUNT // args.decay_step)
        optimizer.zero_grad()
        _, _, logit, prob = model(image)
        predicted = torch.argmax(prob, dim=1)
        acc = torch.Tensor.float(predicted == label).sum() / (label.size(0))
        loss = model.loss(logit, label)
        loss.backward()
        # if args.debug:
        #    pdb.set_trace()
        optimizer.step()
        STEP_COUNT += 1
        acc_loss.append(loss.data.cpu().numpy())
        acc_acc.append(acc.cpu().numpy())
        if ma_acc is None:
            ma_acc = acc_acc[-1]
        else:
            ma_acc = ma_acc * 0.9 + acc_acc[-1] * 0.1
        if ma_loss is None:
            ma_loss = acc_loss[-1]
        else:
            ma_loss = ma_loss * 0.9 + acc_loss[-1] * 0.1
        if len(acc_acc) % args.log_step == 0:
            lapse = time.time() - t
            t = time.time()
            for output in OSTREAM:
                print("\tStep {:0>4d} done, ma loss={:.5f}, ma acc={:.5f}, speed={:.5f} sec/step".format(Iter + 1, ma_loss, ma_acc,
                                                                                                         lapse / args.log_step
                                                                                                         ),
                      file=output)
        if args.debug and Iter > 50:
            break

    print("\n\tCurrent epoch done, average loss={:.5f}, average accuracy={:.5f}\n".format(sum(acc_loss) / len(acc_loss),
                                                                                          sum(acc_acc) / len(acc_acc)
                                                                                          )
          )


def eval(model, train_data, valid_data, test=False):
    acc_acc = []
    mv_stat_update = 20 if not args.test else 100
    if args.bn:
        model.train()
        for Iter, (image, label) in enumerate(train_data):
            _, _, logit, prob = model(image)
            if Iter > mv_stat_update:
                break
    model.eval()
    for Iter, (image, label) in enumerate(valid_data):
        # if args.debug:
        #    pdb.set_trace()
        _, _, logit, prob = model(image)
        predicted = torch.argmax(prob, dim=1)
        acc = torch.Tensor.float(predicted == label).sum() / (label.size(0))
        acc_acc.append(acc.cpu().numpy())
        if args.debug and Iter > 50:
            break
    aver_acc = sum(acc_acc) / len(acc_acc)
    for output in OSTREAM:
        print("Test" if test else "Valid", end="", file=output)
        print(" set, loss={}, acc={:.5f}".format("-",
                                                 aver_acc),
              file=output)
    return aver_acc


class Model(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layer=1, bidirection=False,
                 bias=True, act="relu", batch_norm=True,
                 hidden_min_abs=0, hidden_max_abs=2,
                 gradient_clip=None, init_ih="uniform/-0.001/0.001",
                 cuda=True, bidirection_add=False,
                 multi_layer_add=False, last_hidden=True,
                 debug=args.debug, batch_norm_momentum=0.9
                 ):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.bidirection = bidirection
        self.bias = bias
        self.act = act
        self.batch_norm = batch_norm
        self.hidden_min_abs = hidden_min_abs
        self.hidden_max_abs = hidden_max_abs
        self.gradient_clip = gradient_clip
        self.init_ih = init_ih
        self.bidirection_add = bidirection_add
        self.multi_layer_add = multi_layer_add
        self.last_hidden = last_hidden
        self._cuda = cuda
        self.num_direction = 2 if self.bidirection else 1
        self.num_class = 10
        if not args.mul_label:
            recurrent_initializers = []
            for _ in range(self.num_layer - 1):
                recurrent_initializers.append(lambda weight: nn.init.uniform_(weight, 0, hidden_max_abs))
            recurrent_initializers.append(lambda weight: nn.init.uniform_(weight, hidden_min_abs, hidden_max_abs))
        else:
            recurrent_initializers = None
        RNN = indRNN.BasicIndRNN if args.bn else indRNN.BasicIndRNNLayerNorm
        self.rnn = RNN(input_size=input_size, hidden_size=hidden_size,
                       num_layer=num_layer, bidirection=bidirection,
                       bias=bias, act=act,
                       hidden_max_abs=hidden_max_abs,
                       gradient_clip=gradient_clip, init_ih=init_ih,
                       bidirection_add=bidirection_add,
                       recurrent_weight_initializers=recurrent_initializers,
                       debug=debug, batch_norm_momentum=batch_norm_momentum)

        feature_size = self.hidden_size * (int(self.last_hidden) + (1 - int(self.last_hidden)) * (
            self.num_layer * (1 - int(self.multi_layer_add)) + 1 * int(self.multi_layer_add))) * \
            (self.num_direction - int(self.bidirection and self.bidirection_add))
        if args.dropout:
            self.predictor = nn.Sequential(
                nn.Linear(feature_size,
                          64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, self.num_class),
            )
            nn.init.xavier_uniform_(self.predictor[0].weight, gain=1)
            nn.init.constant_(self.predictor[3].bias, val=0.)
            nn.init.xavier_uniform_(self.predictor[0].weight, gain=1)
            nn.init.constant_(self.predictor[3].bias, val=0.)
        else:

            self.predictor = nn.Linear(feature_size,
                                       self.num_class)
            nn.init.xavier_uniform_(self.predictor.weight, gain=1)
            nn.init.constant_(self.predictor.bias, val=0.)
        self.softmax = nn.Softmax(dim=-1)
        if self._cuda:
            self.predictor.cuda()
            self.rnn.cuda()
            self.softmax.cuda()

    def forward(self, Input):
        batch_size = Input.size(0)
        if len(Input.size()) == 2:
            Input = torch.stack([Input], dim=-1)
        o, h = self.rnn(Input)
        h = h.transpose(0, 1).contiguous()
        if self.last_hidden:
            hn = o[:, -1, :]
        else:
            if self.multi_layer_add:
                if self.bidirection_add:
                    hn = torch.mean(h, dim=1)
                else:
                    hn = torch.cat([torch.mean(h[:, 0::2, :], dim=1),
                                    torch.mean(h[:, 1::2, :], dim=1)], -1)
            else:
                hn = h.view(batch_size, -1)
        logit = self.predictor(hn)
        prob = self.softmax(logit)
        return o, hn, logit, prob  # concate; another option is adding
        # return as batch* hidden_size_s

    def loss(self, pred, label):
        """
        :param pred:batch*num_class
        :param label: (batch,) when kd is off; (batch,num_class) when kd is on
        :return:
        """
        return F.cross_entropy(pred, label)

    def train_network(self):
        if args.weight_decay > 0:
            parameters_decay = []
            parameters_no_decay = []
            for name, param in self.named_parameters():
                if 'weight_hh' in name:  # or 'bias' in name:
                    parameters_no_decay.append(param)
                    # print('parameters no weight decay: ',name)
                else:
                    parameters_decay.append(param)
                    # print('parameters with weight decay: ',name)
            parameters = [
                {"params": parameters_no_decay},
                {"params": parameters_decay, "weight_decay": args.weight_decay}
            ]
        else:
            parameters = self.parameters()
        if args.optim == "rms":
            optimizer = torch.optim.RMSprop(parameters, lr=args.learning_rate, momentum=args.momentum)
        elif args.optim == "adam":
            optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
        elif args.optim == "sgd":
            optimizer = torch.optim.SGD(parameters, lr=args.learning_rate, momentum=args.momentum)
        else:
            raise ValueError("non-existent optimizer")
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate, last_epoch=-1)
        epoch = args.epochs
        valid_acc = [0]
        test_acc = None
        for output in OSTREAM:
            print("Start to train from scratch", file=output)
        for e in range(epoch):
            for output in OSTREAM:
                print("Epoch {:>3d}/{} starts:".format(e + 1, epoch), file=output)
            train(self, data.generator("train", args.batch_size), optimizer, lr_scheduler)
            valid_acc.append(eval(self, data.generator("train", args.batch_size * 4), data.generator("valid", args.batch_size)))
            for output in OSTREAM:
                print(
                    f"Current best results: valid acc={max(valid_acc)}\n",
                    file=output)
            if valid_acc[-1] > max(valid_acc[:-1]):
                self.SaveModel()
                print("Model saved")
            if valid_acc[-1] > 0.99:
                break

        if test_acc is None:
            self.LoadModel()
            test_acc = eval(self, data.generator("train", args.batch_size * 4), data.generator("test", args.batch_size), True)

        for output in OSTREAM:
            print("Model trained:", file=output)
            print(f"Valid acc={valid_acc[-1]}, Test acc={test_acc}", file=output)
        save_model(self)

    def LoadModel(self, path=None):
        if path:
            self.load_state_dict(load_model(path))
        else:
            self.load_state_dict(load_model())
        print("Model loaded")

    def SaveModel(self):
        save_model(self, is_best=True)


def save_model(model, path=os.path.join(args.log_folder, args.save), is_best=False):
    torch.save(model.state_dict(), path)
    if is_best:
        shutil.copy(path, "/".join(path.split('/')[:-1]) + '/best_' + path.split('/')[-1])


def load_model(path=os.path.join(args.log_folder, args.save)):
    return torch.load(path)


def run():
    pass


if __name__ == '__main__':
    data = DataLoader(args.dataset)
    model = Model(input_size=args.inpd, hidden_size=args.hidden_size,
                  num_layer=args.num_layer, bidirection=args.bid,
                  bias=args.bias, act=args.act, batch_norm=args.bn,
                  hidden_min_abs=hidden_min_abs, hidden_max_abs=hidden_max_abs,
                  gradient_clip=(args.min_grad, args.max_grad) if args.g_clip else None, init_ih=args.init_ih,
                  cuda=CUDA, bidirection_add=args.bidadd,
                  multi_layer_add=args.mutadd, last_hidden=args.last_hidden,
                  debug=args.debug, batch_norm_momentum=args.momentum
                  )
    epoch = args.epochs

    if not args.test:
        model.train_network()  # bug: did not enter kd mode
    else:
        assert os.path.isfile(args.test_model) and (args.dataset != "pmnist" or os.path.isfile(args.permutator))
        model.LoadModel(args.test_model)
        valid_acc = eval(model, data.generator("train", args.batch_size * 4), data.generator("valid", args.batch_size))
        test_acc = eval(model, data.generator("train", args.batch_size * 4), data.generator("test", args.batch_size),
                        True)
        print(f"Valid acc={valid_acc}\nTest acc={test_acc}")

    if LOG:
        LOG.close()
