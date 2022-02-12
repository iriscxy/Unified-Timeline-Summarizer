import math

import os

c = 0
r = 0


def getCAndR(candidateSentence, referenceSentences):
    global c
    global r
    referenceCount = []
    referenceLength = []
    c += len(candidateSentence)
    for index3 in range(0, len(referenceSentences)):
        referenceCount.append(abs(len(referenceSentences[index3]) - len(candidateSentence)))
        referenceLength.append(len(referenceSentences[index3]))
    r += referenceLength[referenceCount.index(min(referenceCount))]


def getBP():
    if c > r:
        return 1
    else:
        return math.exp(1 - r / float(c))


def getFiles(candidatePath, referencePath):
    candidatefile = candidatePath
    referencefiles = []
    if os.path.isfile(referencePath):
        referencefiles.append(referencePath)
    else:
        referencefiles = os.listdir(referencePath)
        for i in range(0, len(referencefiles)):
            referencefiles[i] = referencePath + "/" + referencefiles[i]
    return candidatefile, referencefiles


def readFiles(candidatefile, referencefiles):
    candidateData = []
    referencesData = []
    with open(candidatefile) as fp:
        for line in fp:
            candidateData.append(line.strip())
    for i in range(0, len(referencefiles)):
        temp = []
        with open(referencefiles[i]) as fp:
            for line in fp:
                temp.append(line.strip())
        referencesData.append(temp)
    return candidateData, referencesData


def uniGramDictionary(sentence):
    dictionary = {}
    for i in range(0, len(sentence)):

        unigram = sentence[i]
        if unigram in dictionary:
            dictionary[unigram] += 1
        else:
            dictionary[unigram] = 1
    return dictionary


def biGramDictionary(sentence):
    dictionary = {}
    for i in range(0, len(sentence)):

        if i == len(sentence) - 1:
            break
        else:
            bigram = sentence[i] + " " + sentence[i + 1]
            if bigram in dictionary:
                dictionary[bigram] += 1
            else:
                dictionary[bigram] = 1
    return dictionary


def triGramDictionary(sentence):
    dictionary = {}
    if len(sentence) < 2:
        return dictionary
    for i in range(0, len(sentence)):
        # print i
        if i == len(sentence) - 2:
            break
        else:
            trigram = sentence[i] + " " + sentence[i + 1] + " " + sentence[i + 2]
            if trigram in dictionary:
                dictionary[trigram] += 1
            else:
                dictionary[trigram] = 1
    return dictionary


def quadrupleGramDictionary(sentence):
    dictionary = {}
    if len(sentence) < 3:
        return dictionary
    for i in range(0, len(sentence)):

        if i == len(sentence) - 3:
            break
        else:
            quadruplegram = sentence[i] + " " + sentence[i + 1] + " " + sentence[i + 2] + " " + sentence[i + 3]
            if quadruplegram in dictionary:
                dictionary[quadruplegram] += 1
            else:
                dictionary[quadruplegram] = 1
    return dictionary


def uniGram(candidateSentence, referenceSentences):
    referenceDict = []
    reference = []
    candidateSentence = candidateSentence.lower().split()
    candidateSentence = list(filter(lambda x: x is not None, candidateSentence))
    candidateDict = uniGramDictionary(candidateSentence)
    count = 0
    for line in referenceSentences:
        line = line.lower().split()
        line = list(filter(lambda x: x is not None, line))
        reference.append(line)
        referenceDict.append(uniGramDictionary(line))
    getCAndR(candidateSentence, reference)
    for word in candidateDict:
        maxRefIndex = 0
        for index2 in range(0, len(referenceDict)):
            if word in referenceDict[index2]:
                maxRefIndex = max(maxRefIndex, referenceDict[index2][word])

        count += min(candidateDict[word], maxRefIndex)

    return count, len(candidateSentence)


def biGram(candidateSentence, referenceSentences):
    referenceDict = []
    candidateSentence = candidateSentence.lower().split()
    candidateSentence = list(filter(lambda x: x is not None, candidateSentence))
    candidateDict = biGramDictionary(candidateSentence)
    count = 0
    for line in referenceSentences:
        line = line.lower().split()
        line = list(filter(lambda x: x is not None, line))
        referenceDict.append(biGramDictionary(line))
    for word in candidateDict:
        maxRefIndex = 0
        for index2 in range(0, len(referenceDict)):
            if word in referenceDict[index2]:
                maxRefIndex = max(maxRefIndex, referenceDict[index2][word])
        count += min(candidateDict[word], maxRefIndex)
    sumngram = 0
    for values in candidateDict.values():
        sumngram += values
    return count, sumngram


def triGram(candidateSentence, referenceSentences):
    referenceDict = []
    candidateSentence = candidateSentence.lower().split()
    candidateSentence = list(filter(lambda x: x is not None, candidateSentence))
    candidateDict = triGramDictionary(candidateSentence)
    count = 0
    for line in referenceSentences:
        line = line.lower().split()
        line = list(filter(lambda x: x is not None, line))
        referenceDict.append(triGramDictionary(line))
    for word in candidateDict:
        maxRefIndex = 0
        for index2 in range(0, len(referenceDict)):
            if word in referenceDict[index2]:
                maxRefIndex = max(maxRefIndex, referenceDict[index2][word])

        count += min(candidateDict[word], maxRefIndex)
    sumngram = 0
    for values in candidateDict.values():
        sumngram += values
    return count, sumngram


def quadrupleGram(candidateSentence, referenceSentences):
    referenceDict = []
    candidateSentence = candidateSentence.lower().split()
    candidateSentence = list(filter(lambda x: x is not None, candidateSentence))
    candidateDict = quadrupleGramDictionary(candidateSentence)
    count = 0
    for line in referenceSentences:
        line = line.lower().split()
        line = list(filter(lambda x: x is not None, line))
        referenceDict.append(quadrupleGramDictionary(line))
    for word in candidateDict:
        maxRefIndex = 0
        for index2 in range(0, len(referenceDict)):
            if word in referenceDict[index2]:
                maxRefIndex = max(maxRefIndex, referenceDict[index2][word])
        count += min(candidateDict[word], maxRefIndex)
    sumngram = 0
    for values in candidateDict.values():
        sumngram += values
    return count, sumngram


def getModifiedPrecision(candidateData, referencesData):
    uniNum = 0
    uniDen = 0
    biNum = 0
    biDen = 0
    triNum = 0
    triDen = 0
    quadrupleNum = 0
    quadrupleDen = 0
    for index in range(0, len(candidateData)):
        referenceSentences = []
        candidateSentence = candidateData[index]
        for index1 in range(0, len(referencesData)):
            referenceSentences.append(referencesData[index1][index])
        uniClipCount, uniCount = uniGram(candidateSentence, referenceSentences)
        uniNum += uniClipCount
        uniDen += uniCount
        biClipCount, biCount = biGram(candidateSentence, referenceSentences)
        biNum += biClipCount
        biDen += biCount
        triClipCount, triCount = triGram(candidateSentence, referenceSentences)
        triNum += triClipCount
        triDen += triCount
        quadrupleClipCount, quadrupleCount = quadrupleGram(candidateSentence, referenceSentences)
        quadrupleNum += quadrupleClipCount
        quadrupleDen += quadrupleCount
    # print uniNum,uniDen
    # print biNum,biDen
    # print triNum,triDen
    # print quadrupleNum,quadrupleDen
    unigram1 = uniNum / float(uniDen)
    bigram1 = biNum / float(biDen)
    trigram1 = triNum / float(triDen)
    quadruplegram1 = quadrupleNum / float(quadrupleDen)
    # print unigram1,bigram1,trigram1,quadruplegram1
    bleu = getBP() * math.exp(
        0.25 * math.log(unigram1) + 0.25 * math.log(bigram1) + 0.25 * math.log(trigram1) + 0.25 * math.log(
            quadruplegram1))

    bleu_1 = unigram1
    bleu_2 = bigram1
    bleu_3 = trigram1
    bleu_4 = quadruplegram1
    return 100 * bleu, 100 * bleu_1, 100 * bleu_2, 100 * bleu_3, 100 * bleu_4


def calcu_bleu(candidates, references):
    # candidatefile,referencefiles = getFiles(sys.argv[1],sys.argv[2])
    candidatefile, referencefiles = getFiles(candidates, references)

    candidateData, referencesData = readFiles(candidatefile, referencefiles)
    bleu, bleu_1, bleu_2, bleu_3, bleu_4 = getModifiedPrecision(candidateData, referencesData)
    # print c,r
    return bleu, bleu_1, bleu_2, bleu_3, bleu_4
