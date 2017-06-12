__author__ = 'fanfan'
#第一步，将从数据库里面导出来的cvs文件转化为分词以后的txt文件

from setting import tv_classfication
import os
import csv
import jieba



def prepare_data():
    #(path,dirs,files)  = os.walk(tv_classfication.tv_data_path)
    for path,dirs,files in os.walk(tv_classfication.tv_data_path):
        if len(files) == 0:
            continue
        for file in files:
            if file.endswith(".txt"):
                continue
            file_path = os.path.join(path,file)

            write_file_path = file_path.replace(".csv",".txt")
            fp_write = open(write_file_path,'wb')
            first = 1
            with open(file_path,'rb') as csv_reader:
                for line in csv_reader:
                    if first == 1:
                        first = 0
                        continue
                    line = line.decode("utf-8")
                    line = line.replace('"',"")
                    if line =="":
                        continue
                    line = line.split(";")[0]
                    line = jieba.cut(line)
                    line = " ".join(line)+ "\n"
                    fp_write.write(line.encode("utf-8") )
            fp_write.close()


def prepare_data_thu():
    store_path = os.path.dirname(tv_classfication.tv_data_path)

    write_file_path = os.path.join(store_path,"thu_jieba.txt")
    fp_write = open(write_file_path,'wb')
    for path,dirs,files in os.walk(tv_classfication.thc_data_path):
        if len(files) == 0:
            continue
        file_count = 0
        print("doing path:%s" % path)
        for file in files:
            file_count += 1
            file_path = os.path.join(path,file)
            print("doing path:%s, file:%d" % (path,file_count))

            if file_count <10000:
                with open(file_path,'rb') as reader:
                    for line in reader:
                        line = line.decode("utf-8")
                        line = line.replace(' ',"").replace("\n","").replace("\t","").strip()
                        line = jieba.cut(line)
                        line = " ".join(line)+ "\n"
                        fp_write.write(line.encode("utf-8") )
            else:
                break

    fp_write.close()








if __name__ == '__main__':
    #prepare_data()
    prepare_data_thu()