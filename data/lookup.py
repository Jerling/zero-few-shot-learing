import glob
import csv
import scipy.io as sio

def getstruct(dir_ = '.'):
    dic = {}
    files = glob.glob("".join(dir_+'/*/*.mat'))
    #  print(files)
    for f in files:
        #  print(f)
        k = sio.loadmat(f)
        #  print(type(k))
        #  print(k.keys())
        dic[''.join(f)] = list(k.keys())[3:]
    return dic


if __name__ == '__main__':
    dic = getstruct()

    with open("struct.csv","w") as f:
        writer = csv.writer(f)
        for key, value in dic.items():
            writer.writerow([key, value])
