import json

csvFileName = "behaviourData.csv"
jsonDataFileName = "behaviourData.json"
jsonTransitionFileName = "transitionData.json"

jsonDataList = []
jsonTransitionList = []
with open(csvFileName, "r") as f:
    # Read lines in as a generator to conserve space with massive files
    fileLines = f.readlines()
    
    # Loop through the lines
    for line in fileLines:

        # Convert lines into lists of items
        items = line.split(",")

        # Ensuring that we're dealing with the correct kind of data
        # "anon" is the name given when collecting data and the script isn't running
        if len(items) == 68 and items[-1].strip() not in ["anon\n", "anon"]:

            # Seperating temperatures from the metadata
            temps = items[:64]
            metadata = items[64:]

            # Turning the temps into floats
            temps = list(map(float, temps))

            # Reshaping the data into an 8x8 grid (yes I realise this is bad code, but it's fast)
            temps = [temps[:8], temps[8:16], temps[16:24], temps[24:32], temps[32:40], temps[40:48], temps[48:56], temps[56:64]] 
            #print(temps)
            print(metadata)

            entryDict = {"temps": temps,
                         "state": metadata[0],
                         "name": metadata[3],
                         "time": float(metadata[1])}
            
            # Saving the transition data separately.
            if entryDict["state"] == "transition":
                jsonTransitionList.append(entryDict)
            else:
                jsonDataList.append(entryDict)

        else:
            print("yeanahrip")

# Saving the main data file
with open(jsonDataFileName, 'w') as dataOutFile:
    json.dump(jsonDataList, dataOutFile)

# Saving the transition data file
with open(jsonTransitionFileName, 'w') as transitionOutFile:
    json.dump(jsonTransitionList, transitionOutFile)