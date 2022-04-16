from Table import Table

def main():
    key = input("press 0 if you want the table of the article as result, press 1 if you want test something else\n")
    trainDomain = []
    testDomain = []
    ngram = input('press 1 if you want to use unigrams, press 2 if you want to use bigrams, press 3 if you want to use trigrams\n')
    tf_idf = input('press 1 if you want to use tf_idf, 0 otherwise\n')
    if (key == '0'):
        domain = ['Birth Control', 'Depression', 'Pain', 'Anxiety', 'Diabetes, Type 2']
        t = Table(domain, domain, ngram, tf_idf)
        print(t.accuracytable())
    elif (key == '1'):
        domainNumber = int(input("insert the number of domain to train\n"))
        print('train domain')
        for i in range(domainNumber):
            trainDomain.append(input())
        key = input("insert the number of domain to test\n")
        domainNumber = int(key)
        print('test domain')
        for i in range(domainNumber):
            testDomain.append(input())
        t = Table(trainDomain, testDomain, ngram, tf_idf)
        print(t.accuracytable())

if __name__ == '__main__':
    main()










