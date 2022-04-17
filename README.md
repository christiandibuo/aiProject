# aiProject
### Descrizione del ruolo dei moduli sorgente
main.py: è il modulo da cui parte l'esecuzione, permette di scegliere se utilizzare unigram, bigram o trigram per la tokenizzazione del testo, è inoltre possibile dare una priorità maggiore agli n-gram che compaiono meno frequentemente nel testo utilizzando la classe tf-idf di scikit-learn.

BagOfWord.py: essenzialmente è il modulo che contiene la trasformazione in n-gram

Percettrone.py: effettua il training ed il testing del perceptron e verfica poi l' accuratezza della predizione attraverso la classe Perceptron di scikit-learn

### sequenza di comandi che permette di riprodurre i risultati
Per riprodurre i risultati è necessario eseguire il modulo main.py e seguire le indicazioni a schermo.
In particolare se si vuole ottenere la table 3 dell'articolo Gräßer et al. 2018 è necessario scegliere l'opzione bigram quando verrà proposta la scelta, in quanto con gli unigram la classificazione non sarebbe abbastanza accurata.
È possibile generare una tabella simile con dei domini personalizzati, differenti da quelli dell'articolo. 
