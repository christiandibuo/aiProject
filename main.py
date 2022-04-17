from Percettrone import Percettrone

def main():
    key = input("press 0 if you want the table of the article as result, press 1 if you want test something else\n")
    trainDomain = []
    testDomain = []
    ngram = input('press 1 if you want to use unigram, press 2 if you want to use bigram, press 3 if you want to use trigram\n')
    tf_idf = input('press 1 if you want to use tf_idf, 0 otherwise\n')
    if (key == '0'):
        domain = ['Birth Control', 'Depression', 'Pain', 'Anxiety', 'Diabetes, Type 2']
        t = Percettrone(domain, domain, ngram, tf_idf)
        print(t.accuracy())
    elif (key == '1'):
        while True:
            try:
                trainNumber = int(input("insert the number of domain with which train the algorithm\n"))
                break
            except ValueError:
                print("Insert an integer number")
        print('insert the train domain')
        for i in range(trainNumber):
            trainDomain.append(input())
        while True:
            try:
                testNumber = int(input("insert the number of domain to test\n"))
                break
            except ValueError:
                print("Insert an integer number")
        print('insert the test domain')
        for i in range(testNumber):
            testDomain.append(input())
        t = Table(trainDomain, testDomain, ngram, tf_idf)
        print(t.accuracy())

if __name__ == '__main__':
    main()










