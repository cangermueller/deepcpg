import numpy as np


# TODO: Implement nb_sample
def data_generator(data_files, batch_size=128, nb_sample=None, targets=None, shuffle=True, loop=True):
    file_idx = 0
    data_files = list(data_files)
    if nb_sample is None:
        nb_sample = np.inf
    while True:
        if shuffle and file_idx == 0:
            np.random.shuffle(data_files)
            nb_seen = 0
        data_file = h5.File(data_files[file_idx], 'r')
        nb_sample_file = len(data_file['pos'])
        nb_batch = int(np.ceil(nb_sample_file / batch_size))
        for batch in range(nb_batch):
            batch_start = batch * batch_size
            batch_end = min(nb_sample_file, batch_start + batch_size)
            nb_seen += batch_end - batch_start
            if nb_seen > nb_sample:
                data_files = data_files[:file_idx]



            xs = []
            if 'dna' in data_file:
                xs.append(data_file['dna'][batch_start:batch_end])

            ys = []
            ws = []
            target_names = list(data_file['cpg'].keys())
            if targets is not None:
                if isinstance(targets, list):
                    target_names = [target for target in target_names if target in targets]
                else:
                    target_names = target_names[:targets]
            for target in target_names:
                y = data_file['cpg'][target][batch_start:batch_end]
                w = get_sample_weights(y)
                ys.append(y)
                ws.append(w)

            yield (xs, ys, ws)
        file_idx += 1
        if file_idx >= len(data_files):
            file_idx = 0

