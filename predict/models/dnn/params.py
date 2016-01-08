import yaml
import numpy as np
import re


class CpgParams(object):

    def __init__(self):
        self.activation = 'relu'
        self.nb_filter = 4
        self.filter_len = 4
        self.pool_len = 2
        self.nb_hidden = 32
        self.drop_in = 0.0
        self.drop_out = 0.2
        self.batch_norm = False

    def validate(self):
        self.pool_len = min(self.pool_len, self.nb_filter)

    def update(self, params):
        self.__dict__.update(params)

    def __str__(self):
        params = vars(self)
        s = ''
        for k in sorted(params.keys()):
            s += '%s: %s\n' % (k, str(params[k]))
        return s.strip()


class SeqParams(object):

    def __init__(self):
        self.activation = 'relu'
        self.nb_filter = 4
        self.filter_len = 8
        self.pool_len = 4
        self.nb_hidden = 32
        self.drop_in = 0.0
        self.drop_out = 0.2
        self.batch_norm = False

    def validate(self):
        self.pool_len = min(self.pool_len, self.nb_filter)

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
        self.nb_hidden = 16
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


class Params(object):

    def __init__(self):
        self.seq = SeqParams()
        self.cpg = CpgParams()
        self.target = TargetParams()

        self.optimizer = 'Adam'
        self.optimizer_params = {'lr': 0.001}

    def validate(self, nb_hidden=False):
        for k in ['seq', 'cpg', 'target']:
            if hasattr(vars(self)[k], 'validate'):
                vars(self)[k].validate()
        if nb_hidden:
            t = self.target.nb_hidden
            for k in ['seq', 'cpg']:
                s = vars(self)[k]
                if hasattr(s, 'nb_hidden'):
                    t = min(t, s.nb_hidden)
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
            if k in ['seq', 'cpg', 'target']:
                if isinstance(v, dict):
                    t = k.capitalize() + 'Params'
                    vself[k] = globals()[t]()
                    vself[k].update(params[k])
                else:
                    vself[k] = v
            elif k in vself.keys():
                vself[k] = v

    def __str__(self):
        s = 'Seq model:\n'
        s += '---------\n'
        s += str(self.seq)
        s += '\n\nCpG model:\n'
        s += '----------\n'
        s += str(self.cpg)
        s += '\n\nTarget model:\n'
        s += '-------------\n'
        s += str(self.target)
        s += '\n'

        params = vars(self)
        for k in sorted(params.keys()):
            if k not in ['seq', 'cpg', 'target']:
                s += '\n%s: %s' % (k, params[k])
        return s


def sample_dict(param_dist):
    sample = dict()
    for k, v in param_dist.items():
        if isinstance(v, dict):
            sample[k] = sample_dict(v)
        elif isinstance(v, list):
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
            for s in ['cpg', 'seq', 'target']:
                sub = vars(param)[s]
                if hasattr(sub, '__dict__') and k in vars(sub):
                    vars(sub)[k] = v
        param.validate()

        self._nb_sample += 1
        return param
