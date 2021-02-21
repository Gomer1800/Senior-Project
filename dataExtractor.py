import sys
import os
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
        # exit program if '-clean'  is solely called
        if options.a is None and options.n is None:
            return

    valid_percent = 0.15
    config_file = options.config_file
    data_files = None
    download_type = None

    # Save the file to download the images from
    if options.n is None and options.a is not None:
        data_files = options.a
        download_type = '-a'

    elif options.a is None and options.n is not None:
        data_files = options.n
        download_type = '-n'

    # Save the percentage to use for validation
    elif options.percentage is not None:
        try:
            valid_percent = float(options.percentage)
        except:
            print("Percentage used for the validation set must be a float between 0-1")
            return

    # Not all the required arguments were provided
    if config_file is None or data_files is None or download_type is None:
        parser.print_help()
        return

    # Download the images and their associated data
    if options.api == 'labelbox':
        labelBox.downloadImageData(download_type, data_files[0], config_file)

    elif options.api == 'scaleai':
        scaleAI.downloadImageData(download_type, data_files[0], config_file)

    elif options.api == 'both':
        # Use file extensions to call correct api
        for data_file in data_files:
            root, ext = os.path.splitext(data_file)
            if ext == '.csv':
                labelBox.downloadImageData(download_type, data_file, config_file)
            elif ext == '.json':
                scaleAI.downloadImageData(download_type, data_file, config_file)
            else:
                # TODO(LUIS): Handle error, skipping file for now
                print("Skipping file" + data_file)
    else:
        # TODO(LUIS): throw an error
        return

    # Split images into training and validation directories,
    # Creates new random splits on every call
    print("Splitting images into training and validation")
    generic.splitImages(valid_percent)
    return


if __name__ == '__main__':
    _main()
