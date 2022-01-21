import re
def readfile():
    file_name = open('./1w_rep1.txt')
    write_txt = './1w_rep2.txt'
    with open(write_txt, 'a+') as f:
        for line in file_name.readlines():
            line = line.strip('\n')
            strs = re.findall('\d+SM', line)
            if len(strs) > 0:
                str = strs[0]
                x = re.findall('\d+', str)
                num = int(x[0])
                if num >= 2:
                    f.write(line + '\n')
                    print(line)


if __name__ == '__main__':

    # check()
    readfile()