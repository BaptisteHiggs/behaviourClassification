import json

csvFileName = "behaviourData.csv"
jsonFileName = "behaviourData.json"

jsonList = []
with open(csvFileName, "r") as f:
    # Read lines in as a generator to conserve space with massive files
    fileLines = f.readlines()
    
    # Loop through the lines
    for line in fileLines:

        # Convert lines into lists of items
        items = line.split(",")

        # Ensuring that we're dealing with the correct kind of data
        if len(items) == 68:

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
            
            jsonList.append(entryDict)

        else:
            print("yeanahrip")


with open(jsonFileName, 'w') as outfile:
    json.dump(jsonList, outfile)