import io
import numpy as np

def getTxtAddress(baseAddr, folder, psyModel, baseFileName, numOfDoc):
    addressList = []
    for i in range(numOfDoc):
        num = str(i+1)
        if( i<9) :
            num = '0' + str(i+1)
        addressList.append(baseAddr + '\\'+folder +'\\'+psyModel +'\\'+ num + baseFileName+'.txt')    
    return addressList

def getWriteToAddress(numOfToken, baseAddr, folder, model, numOfDoc):
    addressList = []
    for i in range(numOfDoc):
        num = str(i+1)
        if( i<99 ):
            num = '0' + str(i+1)
        if( i<9) :
            num = '00' + str(i+1)
        addressList.append(baseAddr + '\\' + str(numOfToken) +' Token'+'\\'+ model+'\\'+ folder +'\\'+folder+ '_' +model + num +'.txt')    
    return addressList

def numOfLine(adr):
    count = 0
    f = open( adr, encoding="utf-8")
    for line in f:        
        count +=1
    return count

def readToOneDoc(addrList, baseAddr, folder, psyModel, baseFileName):
    document = ""
    totalLen = 0
    for adr in addrList:  
        lineNum = numOfLine(adr)
        rFile = open( adr, encoding="utf-8")

        for line in rFile: 
            line = line[:-1]
            line += " "
            totalLen  += len(line)    
            document += line 
            wFile = open(baseAddr + '\\'+ folder + '\\'+ psyModel + '\\'+ baseFileName + '.txt', 'w', encoding="utf-8")
            wFile.write(document);           
    return document

def countTokenizer(text):
    text = text.strip()
    countBlank = 0
    i = 0;
    while(i< len(text)):
        
        while (i< len(text) and text[i] != " "):            
            i += 1
        countBlank +=1
        while (i< len(text) and text[i] == " "):
            i += 1            
    return countBlank+1

#devide the whole document to list of lines 
#number of lines = total token/numOfToken
def devideToLines(text, numOfToken):
    i = 0
    count = 1
    line = ""
    lines = []
    
    while(i< len(text)):
        startIndex = 0
        if(text[i] != " "):
            startIndex = i
        while (i< len(text) and text[i] != " "):
            i += 1
        endIndex = i
        # find a word, put in the line
        line += text[startIndex:endIndex]+ " "
        count += 1
        if(count == numOfToken):
            lines.append(line.strip())
            line = ""
            count = 1        
        while (i< len(text) and text[i] == " "):
            i += 1   
    lines.append(line.strip())        
    return lines            



def increaseByduplicate(lines, biggerSize):
    items = list(range(len(lines)))
    np.random.shuffle(items)
    duplication = items[:(biggerSize-len(lines))]
    for i in range(len(duplication)):
        lines.append(lines[duplication[i]])
    return lines


def writeToFile(lines, WriteToaddrList):   
    for j in range(len(lines)):
        wFile = open(WriteToaddrList[j], 'w', encoding="utf-8")
        wFile.write(lines[j]); 
    wFile = open(WriteToaddrList[len(lines)-1], 'w', encoding="utf-8")
    wFile.write(lines[len(lines)-1]);
    print("countTokenizer of lase document: ", countTokenizer(lines[len(lines)-1]))
    return lines

    
numOfToken = 1000
baseAddr_before = r'C:\Users\paint\JupyterNotebook\keras\data set'
baseAddr_after = r'C:\Users\paint\JupyterNotebook\keras\data set after preprocess'
psychosis = 'psychosis'
SMS = 'SMS'

psyModel = ['Psychosis_accurate mode (with punctuation)',
                  'Psychosis_accurate mode (without punctuation)',
                  'Psychosis_search mode (with punctuation)',
                  'Psychosis_search mode (without punctuation)']


psyFileNames = ['_jeiba_accurate_mode_removed_I_P',
                      '_jeiba_accurate_mode_without_punctuation',
                      '_jeiba_search_mode_removed_I_P',
                      '_jeiba_search_mode_without_punctuation']

model = ['accurate_mode_punc',
                    'accurate_mode_without_punc',
                    'search_mode_punc',
                    'search_mode_without_punc']

novelModel = ['talx-P_cut-P_remain',
            'talx-P_cut-P_remove',
            'talx-S_cut-P_remain',
            'talx-S_cut-P_remove']

SMSModel = ['sms-P_cut-P_remain',
           'sms-P_cut-P_remove',
           'sms-S_cut-P_remain',
           'sms-S_cut-P_remove']

#psychosis number of document for each model
psyDocNum = []

#for i in range(len(psyModel)):
#    print('*********', psyModel[i])
#    addrList = getTxtAddress(baseAddr_before,psychosis, psyModel[i], psyFileNames[i],24)
#    document = readToOneDoc(addrList, baseAddr_before, psychosis, psyModel[i], psyFileNames[i])
   

for i in range(len(psyModel)):
    psyAdr = r'C:\Users\paint\JupyterNotebook\keras\data set\Psychosis\\'
    print('#######', psyModel[i])
    psyAdr = psyAdr +  psyModel[i]+'\\'+psyFileNames[i]+'.txt'
    f = open(psyAdr, encoding="utf-8")
    document = f.read()
    
    #devide the whole document to list of lines 
    lines = devideToLines(document, numOfToken)
    print("number of lines ",len(lines))
    
    WriteToaddrList = getWriteToAddress(numOfToken, 
                                        baseAddr_after,
                                        psychosis,
                                        model[i],
                                        len(lines))
    lines = writeToFile(lines, WriteToaddrList)
    psyDocNum.append(len(lines))
    

for i in range(len(SMSModel)):
    SMSAdr = r'C:\Users\paint\JupyterNotebook\keras\data set\SMS\SMS preprocessed data\\'
    print('#######', SMSModel[i])
    SMSAdr = SMSAdr+SMSModel[i]+'.txt'
    f = open(SMSAdr, encoding="utf-8")
    document = f.read()

    #devide the whole document to list of lines 
    lines = devideToLines(document, numOfToken)
    print("number of lines ",len(lines))

    # increase the data set by randomly select and duplicate
    lines = increaseByduplicate(lines, psyDocNum[i])
    
    WriteToaddrList = getWriteToAddress( numOfToken, 
                                        baseAddr_after,
                                        SMS,
                                        model[i],
                                        len(lines))
    lines = writeToFile(lines,WriteToaddrList)

    
def getNovelToAddress(numOfToken, baseAddr, folder, model, numOfDoc):
    addressList = []
    for i in range(numOfDoc):
        num = str(i+1)
        if( i<99 ):
            num = '0' + str(i+1)
        if( i<9) :
            num = '00' + str(i+1)
        addressList.append(baseAddr + '\\' + str(numOfToken) +' Token'+'\\'+ model+'\\'+folder+'_'+ model + num +'.txt')    
    return addressList

baseAddr_novel = r'C:\Users\paint\JupyterNotebook\keras\data set after preprocess\novel data'
for i in range(len(novelModel)):
    novelAdr = r'C:\Users\paint\JupyterNotebook\keras\data set\SMS\Novel preprocessing data\\'
    print('@@@@',novelModel[i])
    novelAdr = novelAdr+novelModel[i]+'.txt'
    f = open(novelAdr, encoding="utf-8")
    document = f.read()
   
    #devide the whole document to list of lines 
    lines = devideToLines(document, numOfToken)
    print("number of lines ",len(lines))
    
    ToAddrList = getNovelToAddress(numOfToken, 
                                        baseAddr_novel,
                                        'Novel',
                                        model[i],
                                        len(lines))
    
    lines = writeToFile(lines,ToAddrList)
