from DependencyTree import DependencyTree

def loadConll(inFile):
    sents = []
    trees = []
    counter = 0
    with open('data/' + inFile, 'r') as fin:
        sentenceTokens = []
        tree = DependencyTree()
        for line in fin:
            line = line.strip()
            line = line.split('\t')
            if len(line) < 10:
                if len(sentenceTokens) > 0:
                    trees.append(tree)
                    sents.append(sentenceTokens)
                    tree = DependencyTree()
                    sentenceTokens = []
            else:
                # Added try block to handle exception encountered in python 3
                try:
                    word = line[1]
                    pos = line[4]
                    head = int(line[6])
                    depType = line[7]

                    token = {}
                    token['word'] = word
                    token['POS'] = pos
                    token['head'] = head
                    token['depType'] = depType
                    sentenceTokens.append(token)
                except:
                    continue
                tree.add(head, depType)

    return sents, trees


def writeConll(outFile, sentences, trees):
    with open(outFile, 'w') as fout:
        for i in range(len(sentences)):
            sent = sentences[i]
            tree = trees[i]
            for j in range(len(sent)):
                fout.write("%d\t%s\t_\t%s\t%s\t_\t%d\t%s\t_\t_\n" % (j+1, sent[j]['word'], sent[j]['POS'], sent[j]['POS'], tree.getHead(j+1), tree.getLabel(j+1)))

            fout.write("\n")



sents, trees = loadConll("train.conll")
print (sents[1])
trees[1].print_tree()

