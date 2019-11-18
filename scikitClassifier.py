# Author of original code: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Majorly modified by: Baptiste Higgs
# License: BSD 3 clause

import matplotlib.pyplot as plt
from sklearn import svm, metrics
import numpy as np
import random

def loadTemps(fileName="behaviourData.csv"):
    with open(fileName, "r") as f:
        # Read lines in as a generator to conserve space with massive files
        fileLines = f.readlines()
        
        # Loop through the lines
        print("Reading in lines...")
        allEntries = []
        for line in fileLines:

            # Convert lines into lists of items
            items = line.split(",")

            # Ensuring that we're dealing with the correct kind of data
            # "anon" is the name given when collecting data and the script isn't running
            if len(items) == 68 and items[-1].strip() not in ["anon\n", "anon"] and items[64].strip() not in ['transition', 'anal_sex', 'vaginal_sex', 'oral_sex', 'masturbate']:

                # Seperating temperatures from the metadata
                temps = items[:64]
                behaviour = items[64]

                # Turning the temps into floats (and divide by 25 to scale to appropriate level)
                temps = list(map(lambda x: float(x), temps))
                allEntries.append([behaviour, temps])
        
        # Shuffle results
        print("Shuffling data...")
        random.shuffle(allEntries)
        
        # Extract the shuffled results:
        allTemps = []
        allBehaviours = []
        for entry in allEntries:
            allBehaviours.append(entry[0])
            allTemps.append(entry[1])
        
        # Conversion from lists to arrays
        allTemps = np.array(allTemps)
        allBehaviours = np.array(allBehaviours)

        # Return shuffled behaviours and temps
        return (allBehaviours, allTemps)


behaviours, temps = loadTemps()
print("Loading data finished!")

trainAmount = round(len(behaviours)*0.85)
#import pdb; pdb.set_trace()

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=10, degree=3, kernel='sigmoid', cache_size=2000, verbose=True)
print("Classifier created. Training...")

# We learn the digits on the first half of the digits
classifier.fit(temps[:trainAmount], behaviours[:trainAmount])

print("Predicting...")
# Now predict the value of the digit on the second half:
expected = behaviours[trainAmount:]
predicted = classifier.predict(temps[trainAmount:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
confMatrix = metrics.confusion_matrix(expected, predicted)

for line in confMatrix:
    string = "["
    for x in line:
        string += str(x) + ", "
    string = string[:-2] + '],'
    print(string)