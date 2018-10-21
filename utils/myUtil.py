import os
import glob
from datetime import datetime


def get_datetime(string=True, y=True, m=True, d=True):
    if string:
        now = datetime.now()
        if y and m and d:
            return '{0:%Y-%m-%d}'.format(now)
        elif y and m:
            return '{0:%Y-%m}'.format(now)
        elif y:
            return '{0:%Y}'.format(now)
        elif m and d:
            return '{0:%m-%d}'.format(now)
        elif m:
            return '{0:%m}'.format(now)
        elif d:
            return '{0:%d}'.format(now)
    return now


def make_dir(path, time=True, numbering=True):
    if time:
        path += get_datetime()

    if numbering:
        i = 1
        tmp = path
        path += "_{0:00d}".format(i)
        while os.path.exists(path):
            i += 1
            path = tmp
            path += "_{0:00d}".format(i)
        os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)
    return path + '/'


def change_filename_serial(path):
    print("Change files serially: ")
    print("Target path: " + path)
    files = glob.glob(path)
    for i, f in enumerate(files):
        f_title, f_ext = os.path.splitext(f)
        os.rename(f, f_title + '_' + '{0:03d}'.format(i) + f_ext)


def file_tree(path):
    flist = []
    for root, dirs, files in os.walk(path):
        root = os.path.relpath(root, path)
        if root == '.':
            root = ''
        flist.append([root, sorted(dirs), sorted(files)])

    def _exec(arg, lvlist):
        root, dirs, files = arg
        dirlen = len(dirs)
        flen = len(files)

        for i, dname in enumerate(dirs):
            nounder = (i == dirlen - 1 and flen == 0)

            print_file('<' + dname + '>', lvlist, nounder)

            under_root = os.path.join(root, dname)
            under_list = []

            for t in flist:
                if t[0] == under_root:
                    under_list.append(t)

            for j, t in enumerate(under_list):
                if nounder and j == len(under_list) - 1:
                    add = [True]
                else:
                    add = [False]

                _exec(t, lvlist + add)

        for i, fname in enumerate(files):
            print_file(fname, lvlist, (i == flen - 1))

    def print_file(fname, lvlist, last):
        t = ''
        # 最上階層でなければ余白あり
        if len(lvlist):
            t += ' '
        # 2階層以上 (top/dir1/dir2...) なら、
        # その階層分の余白または罫線を左側に。
        # その階層が親の最後のアイテムなら空白に、
        # そうでなければ罫線に。
        if len(lvlist) >= 2:
            for b in lvlist[1:]:
                if b:
                    t += '　  '
                else:
                    t += '│  '

        # 1階層以上なら、ファイル用の罫線
        if len(lvlist):
            if last:
                t += '└ '
            else:
                t += '├ '
        # 出力
        print(t + fname)

    _exec(flist.pop(0), [])
