import sys
import Data_Extractor.generic as generic
import Data_Extractor.labelBox as labelBox
import Data_Extractor.scaleAI as scaleAI

'''Extract input and expected output data from the csv file'''


def _main():
    parser = generic.init_argparse()
    options = parser.parse_args()

    num_args = len(sys.argv)
    if num_args == 1:
        parser.print_help()
        return

    if options.clean is True:
        generic.cleanData()
        return

    validPercent = 0.15
    configFile = options.config_file
    data_files = None
    downloadType = None

    # Save the file to download the images from
    if options.N is None and options.A is not None:
        data_files = options.A[0]
        downloadType = '-a'

    elif options.A is None and options.N is not None:
        data_files = options.N[0]
        downloadType = '-n'

    # Save the percentage to use for validation
    elif options.percentage is not None:
        try:
            validPercent = float(options.percentage)
        except:
            print("Percentage used for the validation set must be a float between 0-1")
            return

    # Not all the required arguments were provided
    if configFile is None or data_files is None or downloadType is None:
        parser.print_help()
        return

    # Download the images and their associated data
    # labelBox.downloadImageData(downloadType, dataFile, configFile)
    scaleAI.downloadImageData(downloadType, data_files, configFile)

    # Split images into training and validation directories,
    # Creates new random splits on every call
    print("Splitting images into training and validation")
    generic.splitImages(validPercent)
    return


if __name__ == '__main__':
    _main()
