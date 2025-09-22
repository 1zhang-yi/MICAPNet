import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class EHRdataset(Dataset):
    def __init__(self, discretizer, normalizer, listfile, dataset_dir, return_names=True, period_length=48.0, ehr_pkl_fpath=None):
        self.return_names = return_names
        self.discretizer = discretizer
        self.normalizer = normalizer
        self._period_length = period_length

        self._dataset_dir = dataset_dir
        listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self.CLASSES = self._listfile_header.strip().split(',')[5:]
        self._data = self._data[1:]

        self._data = [line.split(',') for line in self._data]
        self.data_map = {
            mas[3]: {
                'labels': list(map(float, mas[5:])),
                'stay_id': float(mas[2]),
                'time': float(mas[4]),
            }
            for mas in self._data
        }

        self.names = list(self.data_map.keys())

        if ehr_pkl_fpath is not None and os.path.isfile(ehr_pkl_fpath):
            print(f'Loading EHR data from {ehr_pkl_fpath}')
            with open(ehr_pkl_fpath, 'rb') as f:
                self.processed_data = pickle.load(f)
        else:
            print(f'Pre-stored pkl file is not found, loading raw EHR data...')
            self.processed_data = {}
            for k in self.names:
                ret = self.read_by_file_name(k)
                data = ret["X"]
                ts = ret["t"] if ret['t'] > 0.0 else self._period_length
                ys = ret["y"]
                names = ret["name"]
                data = self.discretizer.transform(data, end=ts)[0]
                if (self.normalizer is not None):
                    data = self.normalizer.transform(data)
                ys = np.array(ys, dtype=np.int32) if len(ys) > 1 else np.array(ys, dtype=np.int32)[0]
                self.processed_data[k] = {'data': data, 'ys': ys}

            if ehr_pkl_fpath is not None:
                with open(ehr_pkl_fpath, 'wb') as f:
                    pickle.dump(self.processed_data, f)



    def _read_timeseries(self, ts_filename, time_bound=None):

        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                if time_bound is not None:
                    t = float(mas[0])
                    if t > time_bound + 1e-6:
                        break
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_by_file_name(self, index, time_bound=None):
        t = self.data_map[index]['time'] if time_bound is None else time_bound
        y = self.data_map[index]['labels']
        stay_id = self.data_map[index]['stay_id']
        (X, header) = self._read_timeseries(index, time_bound=time_bound)

        return {"X": X,
                "t": t,
                "y": y,
                'stay_id': stay_id,
                "header": header,
                "name": index}


    def __getitem__(self, index, time_bound=None):
        if isinstance(index, int):
            index = self.names[index]
        ret = self.processed_data[index]
        data, ys = ret['data'], ret['ys']
        data[data > 10] = 0  # some missing data represented as 9999999, causing NaN; Filter them out
        data[data < -10] = 0
        return data, ys

    def __len__(self):
        return len(self.names)


def get_ehr_datasets(discretizer, normalizer, args):
    train_ds = EHRdataset(discretizer, normalizer,
                          listfile=f'{args.ehr_data_dir}/{args.task}/train/train_listfile_with_images.csv',
                          dataset_dir=os.path.join(args.ehr_data_dir, f'{args.task}/train'),
                          ehr_pkl_fpath=args.train_ehr_pkl_fpath_paired)
    test_ds = EHRdataset(discretizer, normalizer,
                         listfile=f'{args.ehr_data_dir}/{args.task}/test/test_listfile_with_images.csv',
                         dataset_dir=os.path.join(args.ehr_data_dir, f'{args.task}/test'),
                         ehr_pkl_fpath=args.test_ehr_pkl_fpath_paired)
    return train_ds, test_ds


def get_data_loader(discretizer, normalizer, args):
    train_ds, test_ds = get_ehr_datasets(discretizer, normalizer, args)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16)
    test_dl = DataLoader(test_ds, args.batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16)

    return train_dl, test_dl


def my_collate(batch):
    x = [item[0] for item in batch]
    x, seq_length = pad_zeros(x)
    targets = np.array([item[1] for item in batch])

    x = torch.from_numpy(np.array(x))
    targets = torch.from_numpy(np.array(targets))
    seq_length = torch.from_numpy(np.array(seq_length))

    return [x, targets, seq_length]


def pad_zeros(arr, min_length=None):
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret), seq_length


def load_discretized_header(discretizer, csv_path):

    ret = []
    with open(csv_path, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
        ret = np.stack(ret)
    header = discretizer.transform(ret)[1].split(',')

    return header