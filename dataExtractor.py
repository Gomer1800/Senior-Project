import sys
import Data_Extractor.generic as generic
import Data_Extractor.labelBox as labelBox

'''Extract input and expected output data from the csv file'''


def _main():
    """
    args = sys.argv

    flags = generic.parseCommandLine(numArgs, args)
    """
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
    dataFile = None
    downloadType = None

    # Save the file to download the images from
    if options.n_csv_file is None and options.a_csv_file is not None:
        dataFile = options.a_csv_file
        downloadType = '-a'

    elif options.a_csv_file is None and options.n_csv_file is not None:
        dataFile = options.n_csv_file
        downloadType = '-n'

    # Save the percentage to use for validation
    elif options.percentage is not None:
        try:
            validPercent = float(options.percentage)
        except:
            print("Percentage used for the validation set must be a float between 0-1")
            return

    # Not all the required arguments were provided
    if configFile is None or dataFile is None or downloadType is None:
        parser.print_help()
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
