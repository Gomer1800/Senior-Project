import sys
import Data_Extractor.generic as generic
import Data_Extractor.labelBox as labelBox

'''Extract input and expected output data from the csv file'''


def _main():
    numArgs = len(sys.argv)
    args = sys.argv

    flags = generic.parseCommandLine(numArgs, args)
    # No flags were given
    if flags == []:
        return

    if '-clean' in flags:
        generic.cleanData()
        return

    validPercent = 0.15
    configFile = None
    dataFile = None
    downloadType = None

    for f in flags:
        index = args.index(f)

        # Save the file to download the images from
        if f == '-n' or f == '-a':
            dataFile = args[index + 1]
            downloadType = f

        # Save the percentage to use for validation
        elif f == '-p':
            try:
                validPercent = float(args[index + 1])
            except:
                print("Percentage used for the validation set must be a float between 0-1")
                return

        # Save the configuration file
        elif f == '-c':
            configFile = args[index + 1]

    # Not all the required arguments were provided
    if configFile is None or dataFile is None or downloadType is None:
        print("Usage: python3 dataExtractor.py -clean | -a <filename.csv> -c <filename> [-p <0-1>] |" + \
              "-n <filename.csv> -c <filename> [-p <0-1>]")
        return

    # Download the images and their associated data
    labelBox.downloadImageData(downloadType, dataFile, configFile)

    # Split images into training and validation directories,
    # Creates new random splits on every call
    print("Splitting images into training and validation")
    generic.splitImages(validPercent)
    return


if __name__ == '__main__':
    _main()
