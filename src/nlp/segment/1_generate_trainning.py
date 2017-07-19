import sys
import os
from setting import nlp_segment


totalLine = 0
longLine = 0

MAX_LEN = 80
totalChars = 0

class Sentence:
    def __init__(self):
        self.tokens = []
        self.chars = 0
        self.tagDict = {"0":"S","1":"B","2":"M","3":"E"}

    def addToken(self,f):
        self.chars += len(f)
        self.tokens.append(f)

    def clear(self):
        self.tokens = []
        self.chars = 0

    # label -1, unknown
    # 0-> 'S'
    # 1-> 'B'
    # 2-> 'M'
    # 3-> 'E'

    def generate_tr_line(self,x):
        for t in self.tokens:
            if len(t) == 1:
                x.append('{}/{}'.format(t[0],self.tagDict["0"]))
            else:
                nn = len(t)
                for i in range(nn):
                    writeString = "{}/{}"
                    if i==0:
                        writeString = writeString.format(t[i],self.tagDict["1"])
                    elif i == (nn-1):
                        writeString = writeString.format(t[i],self.tagDict["3"])
                    else:
                        writeString = writeString.format(t[i],self.tagDict["2"])
                    x.append(writeString)


def processToken(token,sentence,out,end):
    global totalLine
    global longLine
    global totalChars
    global MAX_LEN
    nn = len(token)
    while nn > 0 and token[nn -1] != '/':
        nn = nn -1
    token = token[:nn-1].strip()
    if token != '。':
        sentence.addToken(token)

    if token == '。' or end:
        if sentence.chars > MAX_LEN:
            longLine +=1
        else:
            x = []
            totalChars += sentence.chars
            sentence.generate_tr_line(x)
            nn = len(x)
            for j in range(nn,MAX_LEN):
                x.append("。/S")
            line = " ".join(x)

            out.write(("%s\n" % (line)).encode('utf-8'))
        totalLine +=1
        sentence.clear()


def processLine(line,out):
    line = line.strip()
    nn = len(line)
    seeLeftB = False
    start = 0
    sentence = Sentence()
    try:
        for i in range(nn):
            if line[i] ==" ":
                if not seeLeftB:
                    token = line[start:i]
                    if token.startswith('['):
                        tokenLen = len(token)
                        while tokenLen > 0 and token[tokenLen -1] != "]":
                            tokenLen = tokenLen -1
                        token = token[1:tokenLen -1]
                        ss = token.split(" ")
                        for s in ss:
                            processToken(s,sentence,out,False)
                    else:
                        processToken(token,sentence,out,False)
                    start = i + 1
            elif line[i] == '[':
                seeLeftB = True
            elif line[i] == ']':
                seeLeftB = False
        if start < nn:
            token = line[start:]
            processToken(token,sentence,out,True)
    except:
        pass

if __name__ == "__main__":
    global totalChars
    global longLine
    global totalLine

    rootDir = nlp_segment.people_2014
    outPath = nlp_segment.train_path
    out = open(outPath,'wb')
    for dirName,subdirList,fileList in os.walk(rootDir):

        curDir = os.path.join(rootDir,dirName)
        for file in fileList:
            if file.endswith('.txt'):
                curFile = os.path.join(curDir,file)
                fp = open(curFile,'rb')
                for line in fp.readlines():
                    line = line.decode('utf-8')
                    line = line.strip()
                    processLine(line,out)
                fp.close()
    out.close()
    print("total:%d, long lines:%d, chars:%d" %
              (totalLine, longLine, totalChars))


