def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['type'].lower()

    if dataset_type in ['face', 'retouch']:
         from data.face_dataset import FaceDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset