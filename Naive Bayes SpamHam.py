import csv
from itertools import count
import math

def read_csv(userInput):
    with open(userInput, mode = 'r', encoding='utf-8') as file:
        # Create a Dictionary reader object
        csv_reader = csv.DictReader(file)

        Tspam = []
        Tham = []
        Vspam = []
        Vham = []
        i = 0
        rowCount = 0
        for row in csv_reader:
            rowCount+=1
        rowCount *= .67
        trainingRows = int(rowCount)

        # Moves file pointer back to beginning
        file.seek(0)

        # Creates reader again
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            # Accesses Elements via column header names
            target = row['Target']
            target = target.lower()
            data = row['data']
            data = data.lower()
            

            if i < (trainingRows):
                if "spam" in target:
                    Tspam.append(str(data))

                else:
                    Tham.append(str(data))
            else:
                if "spam" in target:
                    Vspam.append(str(data))
                
                else:
                    Vham.append(str(data))

            i+=1

    return Tspam, Tham, Vspam, Vham

#Counts how many unique words are in the training set
def getUniqueWords (Tspam, Tham):
    #set only allows unique inserts
    uniqueWords = set()
    
    for message in Tspam:
        uniqueWords.update(message.split())
    for message in Tham:
        uniqueWords.update(message.split())
    #returns all unique words to be later used to calculate probabilities
    return uniqueWords

#counts how many times a specific word is in a list of strings
def wordPermutations(listStr, word):
    count = 0
    #for every word segment in sentence    
    for message in listStr:
        #updates currWord for every substring in message
        for currWord in message.split():
            if currWord == word:
                count +=1
    #returns permutaions of the word specified
    return count

#calculates the probability of a word given spam or ham
def probOfWord(listStr, word, sizeUniqueWords):
    #finds how many times a word is in the given list
    wordNum = wordPermutations(listStr, word)

    #will store total amount of words in given list
    totalWords = 0

    for message in listStr:
        totalWords += len(message.split())

    #laplace smoothing +1 in numerator & + unique words in denom
    probability = (wordNum + 1) / (totalWords + sizeUniqueWords)

    #returns prob of word in given list
    return probability

def spamHamClassifier(listStr, Tspam, Tham, Pspam, Pham, uniqueWords):
    #calculate prior probs first
    spamLogProb = math.log(Pspam)
    hamLogProb = math.log(Pham)

    sizeUniqueWords = len(uniqueWords)

    #check every word in the Testing set
    for word in listStr.split():

        #words that arent in the training set are ignored
        if word not in uniqueWords:
            continue
        


        spamProb = probOfWord(Tspam, word, sizeUniqueWords)
        hamProb = probOfWord(Tham, word, sizeUniqueWords)

        #Weighing system to give more weight to words that are more likely to be spam or ham. 
        # If the log odds is greater than 1, it will add .75 of the log odds to the 
        # corresponding probability. This is done to prevent the probabilities 
        # from being too extreme and to give more weight to words that are more 
        # likely to be spam or ham.

        logOdds = math.log(spamProb / hamProb)
        if abs(logOdds) > 1:
            if logOdds > 0:
                spamLogProb += 0.75 * logOdds
            else:
                hamLogProb += 0.5 * (-logOdds)


        #adds the log of the conditional probabilities to the prior probabilities
        spamLogProb += math.log(spamProb)
        hamLogProb += math.log(hamProb) 


    if spamLogProb > hamLogProb:
        return "SPAM"
    else:
        return "HAM"

#validation set checker
def spamHamChecker(Vset, Tspam, Tham, Pspam, Pham, uniqueWords):
    for message in Vset:
        result = spamHamClassifier(message, Tspam, Tham, Pspam, Pham, uniqueWords)
        print(f"This sentence is likely {result} \"{message}\"\n")


#-----------------------------MAIN---------------------------------

userInput = str(input("Enter the path of the csv file with data: "))
Tspam, Tham, Vspam, Vham = read_csv(userInput)

#prior probabilities
Pspam = len(Tspam) / (len(Tspam) + len(Tham))
Pham = len(Tham) / (len(Tspam) + len(Tham))

print(f"The prior probability of a Email being spam is {Pspam}\n")
print(f"The prior probability of a Email being ham is {Pham}\n")


uniqueWords = getUniqueWords(Tspam, Tham)
spamHamChecker(Vspam, Tspam, Tham, Pspam, Pham, uniqueWords)
spamHamChecker(Vham, Tspam, Tham, Pspam, Pham, uniqueWords)

