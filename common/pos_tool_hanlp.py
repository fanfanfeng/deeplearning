# -*-coding:utf-8 -*-
from jpype import *
import jieba.posseg as pseg
startJVM(getDefaultJVMPath(), r"-Djava.class.path=/home/fanfan/hanlp/hanlp-1.3.1.jar:/home/fanfan/hanlp", "-Xms1g", "-Xmx1g")
HanLP = JClass('com.hankcs.hanlp.HanLP')
crfHanlp = JClass('com.hankcs.hanlp.dependency.CRFDependencyParser')
nShort = JClass('com.hankcs.hanlp.seg.NShort.NShortSegment')
nlpSeg = JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')
crfSeg = JClass('com.hankcs.hanlp.seg.CRF.CRFSegment')
crfSegObj = crfSeg()
crfSegObj.enablePartOfSpeechTagging(True)
textToParse = u"呼叫水晶梨"
# 依存句法分析
text = HanLP.parseDependency(textToParse).toString()
text1 = crfHanlp.compute(textToParse).toString()
#text2 = HanLP.segment(textToParse).toString()
#text3 = nlpSeg.segment(textToParse).toString()
#print(text3)
print(crfSegObj)
text4 = crfSegObj.seg(JString(textToParse))
print(type(text4))
print(text4)
print('++++++++++++++++')

#nShort.enablePlaceRecognize(True).enableOrganizationRecognize(True)
#print(nShort.parse(textToParse))
#print('++++++++++++++++')
#print(text2)
print (text1)
shutdownJVM()
f = open("coll.txt",'wb+')
#print(type(text))
print (text)


#f.write(text.encode('utf8'))

f.close()

f = open("coll1.txt",'wb+')
f.write(text1.encode('utf8'))
f.close()

words =pseg.cut(textToParse)
for w in words:
    print(w.word,w.flag)

'''
with open('save.txt','wb+') as f :
    for line in open('dict.txt', 'rb'):
        lineList =line.decode('utf-8').replace('\r\n','').split(' ')
        temp = lineList[1]
        lineList[1] =lineList[2]
        lineList[2] = temp+'\n'
        print (lineList)
        f.write(" ".join(lineList).encode('utf-8'))
'''

