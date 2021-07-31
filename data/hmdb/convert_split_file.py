def convert_split_file(old_filename, new_filename, ext="0"):
    buf = ""
    with open(old_filename, 'r') as old:
        for l in old.readlines():
            name, y = l.strip().split(' ')
            f = name.split('.')[0] + f'_{ext}.mp4'
            buf += f"{f} {y}\n"
    with open(new_filename, 'w') as new:
        new.write(buf)

convert_split_file("split0_master.txt", "split0_master_new.txt")
