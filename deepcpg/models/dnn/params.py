import yaml
import numpy as np
import re
import sys


class ConvParams(object):

    def __init__(self):
        self.nb_filter = [4]
        self.filter_len = [8]
        self.pool_len = [2]
        self.activation = 'relu'
        self.nb_hidden = 0
        self.drop_in = 0.0
        self.drop_out = 0.5
        self.l1 = 0.05
        self.l2 = 0.02
        self.batch_norm = False

    def validate(self):
        for k in ['nb_filter', 'filter_len', 'pool_len']:
            if not isinstance(self.__dict__[k], list):
                self.__dict__[k] = [self.__dict__[k]]

        for i in range(len(self.nb_filter)):
            self.pool_len[i] = min(self.pool_len[i], self.nb_filter[i])

    def update(self, params):
        for k, v in params.items():
            if k in ['nb_filter', 'filter_len', 'pool_len']:
                if not isinstance(v, list):
                    params[k] = [v]
        self.__dict__.update(params)

    def __str__(self):
        params = vars(self)
        s = ''
        for k in sorted(params.keys()):
            s += '%s: %s\n' % (k, str(params[k]))
        return s.strip()


class CpgParams(ConvParams):
    pass


class SeqParams(ConvParams):
    pass


class JointParams(object):

    def __init__(self):
        self.nb_hidden = 128
        self.activation = 'relu'
        self.drop_out = 0.2
        self.batch_norm = False

    def update(self, params):
        self.__dict__.update(params)

    def __str__(self):
        params = vars(self)
        s = ''
        for k in sorted(params.keys()):
            s += '%s: %s\n' % (k, str(params[k]))
        return s.strip()


class TargetParams(object):

    def __init__(self):
        self.activation = 'relu'
        self.nb_hidden = 0
        self.drop_out = 0.25
        self.batch_norm = False

    def update(self, params):
        self.__dict__.update(params)

    def __str__(self):
        params = vars(self)
        s = ''
        for k in sorted(params.keys()):
            s += '%s: %s\n' % (k, str(params[k]))
        return s.strip()


class Params(object):

    def __init__(self):
        self.seq = SeqParams()
        self.cpg = CpgParams()
        self.joint = False
        self.target = TargetParams()

        self.optimizer = 'Adam'
        self.optimizer_params = {'lr': 0.001}
        self.batch_size = 128
        self.loss = None

    def validate(self, nb_hidden=False):
        for k in ['seq', 'cpg', 'joint', 'target']:
            if hasattr(vars(self)[k], 'validate'):
                vars(self)[k].validate()
        if nb_hidden:
            t = sys.maxsize
            for k in ['seq', 'cpg']:
                s = vars(self)[k]
                if hasattr(s, 'nb_hidden') and s.nb_hidden > 0:
                    t = min(t, s.nb_hidden)
            if self.joint and self.joint.nb_hidden > 0:
                t = min(self.joint.nb_hidden, t)
                self.joint.nb_hidden = t
            t = min(self.target.nb_hidden, t)
            self.target.nb_hidden = t

    @staticmethod
    def from_yaml(path):
        p = Params()
        with open(path, 'r') as f:
            t = yaml.load(f.read())
            p.update(t)
        return p

    def to_yaml(self, path):
        dparam = self.__dict__
        with open(path, 'w') as f:
            t = yaml.dump(dparam, default_flow_style=False)
            t = re.subn('!![^\s]+', '', t)[0]
            f.write(t)

    def update(self, params):
        vself = vars(self)
        for k, v in dict(params).items():
            if k in ['seq', 'cpg', 'joint', 'target']:
                if isinstance(v, dict):
                    t = k.capitalize() + 'Params'
                    vself[k] = globals()[t]()
                    vself[k].update(params[k])
                else:
                    vself[k] = v
            else:
                vself[k] = v

    def __str__(self):
        s = 'Seq layer:\n'
        s += '---------\n'
        s += str(self.seq)
        s += '\n\nCpG layer:\n'
        s += '----------\n'
        s += str(self.cpg)
        s += '\n\nJoint layer:\n'
        s += '----------\n'
        s += str(self.joint)
        s += '\n\nTarget layer:\n'
        s += '-------------\n'
        s += str(self.target)
        s += '\n'

        params = vars(self)
        for k in sorted(params.keys()):
            if k not in ['seq', 'cpg', 'joint', 'target']:
                s += '\n%s: %s' % (k, params[k])
        return s


def sample_dict(param_dist):
    sample = dict()
    for k, v in param_dist.items():
        if isinstance(v, dict):
            sample[k] = sample_dict(v)
        elif isinstance(v, list):
            if len(v) > 0:
                if hasattr(v[0], 'rvs'):
                    sample[k] = [x.rvs() for x in v]
                else:
                    sample[k] = v[np.random.randint(0, len(v))]
        elif hasattr(v, 'rvs'):
            sample[k] = v.rvs()
        else:
            sample[k] = v
    return sample


class ParamSampler(object):

    def __init__(self, param_dist, nb_sample=1,
                 global_param=['batch_norm', 'activation']):
        self.param_dist = param_dist
        self.nb_sample = nb_sample
        self.global_param = global_param

    def __iter__(self):
        self._nb_sample = 0
        return self

    def __next__(self):
        if self._nb_sample == self.nb_sample:
            raise StopIteration

        sample = sample_dict(self.param_dist)
        gparam = dict()
        t = dict()
        for k, v in sample.items():
            if k in self.global_param:
                gparam[k] = v
            else:
                t[k] = v
        sample = t

        param = Params()
        param.update(sample)
        if 'batch_norm' in gparam and gparam['batch_norm']:
            param.optimizer = 'sgd'
        for k, v in gparam.items():
            for s in ['cpg', 'seq', 'joint', 'target']:
                sub = vars(param)[s]
                if hasattr(sub, '__dict__') and k in vars(sub):
                    vars(sub)[k] = v
        param.validate()

        self._nb_sample += 1
        return param
