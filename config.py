# Model Config

from yacs.config import CfgNode as CN

_c = CN()
_c.plans = 3
_c.img_h = 128
_c.img_w = 128





_c.TRAIN = CN()
#_c.TRAIN.save_model_to = ''
#_c.TRAIN.load_model_from = '/content/drive/My Drive/Colab Notebooks/facial_landmarks_detection/model_pkl/'
#_c.TRAIN.OPTIMIZER = 'adam'
_c.TRAIN.lr = 0.0005
_c.TRAIN.BATCH_SIZE = 40
_c.TRAIN.SHUFFLE = True



_c.TEST = CN()
_c.TEST.BATCH_SIZE = 40



_c.dataset = CN()
_c.dataset.mat_path = '/content/drive/My Drive/Colab Notebooks/facial_landmarks_detection/AFLW/aflw_data.mat'
_c.dataset.train_path = '/content/drive/My Drive/Colab Notebooks/facial_landmarks_detection/AFLW/train_128_pxl/0_/'
_c.dataset.test_path = '/content/drive/My Drive/Colab Notebooks/facial_landmarks_detection/AFLW/test_128_pxl/0_/'
_c.dataset.mean = [.5,.5,.5]
_c.dataset.std = [.5,.5,.5]


_c.lvl_128 = CN()
_c.lvl_128.inplans = 3
_c.lvl_128.outplans = 18



_c.lvl_64 = CN()
_c.lvl_64.inplans = 18
_c.lvl_64.conv7plans = 36
_c.lvl_64.conv5plans = 54
_c.lvl_64.conv3plans = 108
_c.lvl_64.out =_c.lvl_64.inplans + _c.lvl_64.conv7plans +_c.lvl_64.conv5plans + _c.lvl_64.conv3plans
_c.lvl_64.reduce_to = 108




_c.lvl_32 = CN()
_c.lvl_32.inplans = _c.lvl_64.reduce_to
_c.lvl_32.conv7plans = 72
_c.lvl_32.conv5plans = 108
_c.lvl_32.conv3plans = 216
_c.lvl_32.out =_c.lvl_32.inplans + _c.lvl_32.conv7plans +_c.lvl_32.conv5plans + _c.lvl_32.conv3plans
_c.lvl_32.reduce_to = 306




_c.lvl_16 = CN()
_c.lvl_16.inplans = _c.lvl_32.reduce_to
_c.lvl_16.conv7plans = 144
_c.lvl_16.conv5plans = 216
_c.lvl_16.conv3plans = 432
_c.lvl_16.out = _c.lvl_16.inplans + _c.lvl_16.conv7plans +_c.lvl_16.conv5plans + _c.lvl_16.conv3plans
_c.lvl_16.reduce_to = 512





_c.lvl_8 = CN()
_c.lvl_8.inplans = _c.lvl_16.reduce_to
_c.lvl_8.conv7plans = 288
_c.lvl_8.conv5plans = 432
_c.lvl_8.conv3plans = 864
_c.lvl_8.out =_c.lvl_8.inplans + _c.lvl_8.conv7plans +_c.lvl_8.conv5plans + _c.lvl_8.conv3plans
_c.lvl_8.reduce_to = 980




_c.lvl_16up = CN()
_c.lvl_16up.inplans = _c.lvl_8.reduce_to
_c.lvl_16up.conv7plans = 108
_c.lvl_16up.conv5plans = 216
_c.lvl_16up.conv3plans = 432
_c.lvl_16up.out = _c.lvl_16up.conv7plans +_c.lvl_16up.conv5plans + _c.lvl_16up.conv3plans
_c.lvl_16up.reduce_to = 512





_c.lvl_32up = CN()
_c.lvl_32up.inplans = _c.lvl_16.reduce_to + _c.lvl_16up.reduce_to
_c.lvl_32up.conv7plans = 144
_c.lvl_32up.conv5plans = 216
_c.lvl_32up.conv3plans = 432
_c.lvl_32up.out = _c.lvl_32up.conv7plans +_c.lvl_32up.conv5plans + _c.lvl_32up.conv3plans
_c.lvl_32up.reduce_to = 980





_c.lvl_64up = CN()
_c.lvl_64up.inplans = _c.lvl_32.reduce_to + _c.lvl_32up.reduce_to
_c.lvl_64up.conv7plans = 72
_c.lvl_64up.conv5plans = 108
_c.lvl_64up.conv3plans = 216
_c.lvl_64up.out = _c.lvl_64up.conv7plans +_c.lvl_64up.conv5plans + _c.lvl_64up.conv3plans
_c.lvl_64up.reduce_to = 256




_c.out_features = 21



def get_cfg_defaults():
    return _c.clone()


