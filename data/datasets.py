import os
import numpy as np
from scipy import io
import glob

cur_dir = os.path.abspath('.')
data_dir = os.path.join(cur_dir, 'data')

def dataloader(ds = 'CUB'):
    """dataloader 加载数据集

    :param ds: string
            CUB->CUB_data
            CUB1->CUB1_data
            AwA->AwA_data
            AWA1->AwA1_data
            AwA2->AwA2_data
            APY->APY_data
            SUN->SUN_data
    """
    dataset = {}
    data = os.path.join(data_dir, ds+'_data')
    files = glob.glob(os.path.join(data,'*.mat'))
    for f in files:
        fpath = os.path.join(data, f)
        fn = io.loadmat(fpath)
        for key in fn.keys():
            if "__" not in key:
                dataset[key] = np.array(fn[key])
    return dataset

if __name__ == '__main__':
    datasets = ['CUB','CUB1','AwA','AwA1','AwA2','APY','SUN']
    for d in datasets:
        print('# ', d)
        print('key\t | value')
        dataset = dataloader(d)
        for key, value in dataset.items():
            print(key,'\t | ',value.shape)

