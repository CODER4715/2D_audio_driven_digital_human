import os

import argparse

from sklearn.model_selection import train_test_split



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", help="Root folder of the dataset", default='./part')
    parser.add_argument("--filelists_dir", help="dir of output", default='./filelists')

    args = parser.parse_args()
    data_list =[]
    for root,dirs,files in os.walk(args.data_root):
        for dir in dirs:
            d = os.path.join(root, dir)
            if len(d.split('\\'))<3:
                continue
            # print(d)
            d = d.split(args.data_root)[1]
            d = d[1:]
            d = d.replace('\\', '/')
            data_list.append(d)
    print(len(data_list))

    train_set, val_set = train_test_split(data_list, test_size=0.3, random_state=42)
    # print(train_set)
    print(len(train_set))
    print(len(val_set))
    with open(os.path.join(args.filelists_dir, 'train.txt'), "w") as f:
        for i in train_set:
            f.write(i+'\n')  # 自带文件关闭功能，不需要再写f.close()

    with open(os.path.join(args.filelists_dir, 'val.txt'), "w") as f:
        for i in val_set:
            f.write(i+'\n')  # 自带文件关闭功能，不需要再写f.close()



if __name__=='__main__':
    main()