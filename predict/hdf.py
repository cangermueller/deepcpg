import h5py as h5


PATH_SEP = '/'


def split_path(path, sep=':', first=False):
    t = path.split(sep)
    filename = t[0]
    hpath = ''.join(t[1:])
    if len(hpath) == 0 or hpath[0] != PATH_SEP:
        hpath = PATH_SEP + hpath
    if first and hpath == PATH_SEP:
        item = first_item(filename, hpath)
        if item is None:
            raise IOError('No dataset in %s!' % (hpath))
        hpath = PATH_SEP + item
    return (filename, hpath)


def ls(path, group=None):
    f = h5.File(path)
    if group is None or len(group) == 0:
        group = PATH_SEP
    items = list(f[group].keys())
    f.close()
    return items


def first_item(path, group=None):
    items = ls(path, group)
    if len(items) > 0:
        return items[0]
    else:
        return None
