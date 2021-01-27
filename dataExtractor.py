import urllib.request
import urllib.error
import os
import sys
import shutil
from Data_Extractor.generic import downloadImageData
from PIL import Image
from random import randint

class DataExtractor:
   '''Extract input and expected output data from the csv file'''
   def _main(self):
      numArgs = len(sys.argv)
      args = sys.argv

      flags = self.parseCommandLine(numArgs, args);
      #No flags were given
      if flags == []:
         return

      if '-clean' in flags:
         self.cleanData()
         return

      validPercent = 0.15
      configFile = None
      dataFile = None
      downloadType = None

      for f in flags:
         index = args.index(f)

         #Save the file to download the images from
         if f == '-n' or f == '-a':
            dataFile = args[index+1]
            downloadType = f

         #Save the percentage to use for validation
         elif f == '-p':
            try:
               validPercent = float(args[index+1])
            except:
               print("Percentage used for the validation set must be a float between 0-1")
               return

         #Save the configuration file
         elif f == '-c':
            configFile = args[index+1]

      #Not all the required arguments were provided
      if configFile is None or dataFile is None or downloadType is None:
         print("Usage: python3 dataExtractor.py -clean | -a <filename.csv> -c <filename> [-p <0-1>] |" +\
               "-n <filename.csv> -c <filename> [-p <0-1>]")
         return

      #Download the images and their associated data
      downloadImageData(self, downloadType, dataFile, configFile)


      #Split images into training and validation directories,
      #Creates new random splits on every call
      print("Splitting images into training and validation")
      self.splitImages(validPercent)
      return

   '''Parse the command line to find what flags were given'''
   def parseCommandLine(self, numArgs, args):
      flags = []

      for i in range(numArgs):
         if args[i] == '-clean':
            if numArgs != 2:
               print("Usage: python3 dataExtractor.py -clean | -a <filename.csv> -c <filename> [-p <0-1>] |" +\
                     "-n <filename.csv> -c <filename> [-p <0-1>]")
               return []
            flags.append(args[i])
            return flags

         if args[i] == '-n':
            if '-a' in flags or '-n' in flags:
               print("Usage: python3 dataExtractor.py -clean | -a <filename.csv> -c <filename> [-p <0-1>] |" +\
                     "-n <filename.csv> -c <filename> [-p <0-1>]")
               return []
            flags.append(args[i])

         if args[i] == '-a':
            if '-a' in flags or '-n' in flags:
               print("Usage: python3 dataExtractor.py -clean | -a <filename.csv> -c <filename> [-p <0-1>] |" +\
                     "-n <filename.csv> -c <filename> [-p <0-1>]")
               return []
            flags.append(args[i])

         if args[i] == '-p':
            if '-p' in flags:
               print("Usage: python3 dataExtractor.py -clean | -a <filename.csv> -c <filename> [-p <0-1>] |" +\
                     "-n <filename.csv> -c <filename> [-p <0-1>]")
               return []
            flags.append(args[i])

         if args[i] == '-c':
            if '-c' in flags:
               print("Usage: python3 dataExtractor.py -clean | -a <filename.csv> -c <filename> [-p <0-1>] |" +\
                     "-n <filename.csv> -c <filename> [-p <0-1>]")
               return []
            flags.append(args[i])

      return flags

   '''Remove all the directories and files containing data information'''
   def cleanData(self):
      confirm = "None"
      while (confirm.lower() != 'y' and confirm.lower() != 'n'):
         confirm = input("Are you sure you want to delete all image directories and data? (y/n): ")
      if (confirm == 'n'):
         return

      dirPath = os.getcwd() #Get the current directory path

      #Remove the directories and all the files in them
      try:
         if os.path.isdir(dirPath + '/Input_Images'):
            shutil.rmtree(dirPath + '/Input_Images')
         if os.path.isdir(dirPath + '/Image_Masks'):
            shutil.rmtree(dirPath + '/Image_Masks')
         if os.path.isdir(dirPath + '/Mask_Data'):
            shutil.rmtree(dirPath + '/Mask_Data')
         if os.path.isdir(dirPath + '/Mask_Validation'):
            shutil.rmtree(dirPath + '/Mask_Validation')
         if os.path.isdir(dirPath + '/Blacklist_Masks'):
            shutil.rmtree(dirPath + '/Blacklist_Masks')
         if os.path.isdir(dirPath + '/Whitelist_Masks'):
            shutil.rmtree(dirPath + '/Whitelist_Masks')
         if os.path.isdir(dirPath + '/Training_Images'):
            shutil.rmtree(dirPath + '/Training_Images')
         if os.path.isdir(dirPath + '/Validation_Images'):
            shutil.rmtree(dirPath + '/Validation_Images')
         if os.path.isdir(dirPath + '/Unlabeled'):
            shutil.rmtree(dirPath + '/Unlabeled')
         if os.path.isfile(dirPath + '/Whitelisted_Images.txt'):
            os.remove(dirPath + '/Whitelisted_Images.txt')
         if os.path.isfile(dirPath + '/Blacklisted_Images.txt'):
            os.remove(dirPath + '/Blacklisted_Images.txt')
      except OSError as err:
         print("Error: {0}".format(err))
         return

   '''Extract 128 points representing the bounds of the image mask between 0-1
      Takes in the pixel array representing the mask, and the width & height of the mask'''
   def extractMaskPoints(self, pixels, width, height, numOutputs):
      found = False
      maskData = []
      stepSize = width // numOutputs

      #Find the numOutputs points along the image that represent the boundary of free space
      #Find the boundary goint from bottom (height - 1) to top (0)
      for x in range(0, width, stepSize):
         for y in range(height-1, -1, -1):
            color = pixels[x,y]
            if color == 0:
               break
         maskData.append(y / (height - 1))
         found = False

      return maskData

   '''Download the image from the given URL. Return None if the request fails more than 5 times'''
   def getImageFromURL(self, url):
      image = None
      trys = 0
      #Attempt to download the image 5 times before quitting
      while (trys < 5):
         try:
            return urllib.request.urlretrieve(url) #Retrieve the image from the URL
         except urllib.error.URLError as err:
            print("Error: {0}".format(err))
            print("Trying again")
         except urllib.error.HTTPError as err:
            print("Error: {0}".format(err))
            print("Trying again")
         except urllib.error.ContentTooShortError as err:
            print("Error: {0}".format(err))
            print("Trying again")
         trys += 1
      return None

   '''Return True if there is a black edge along the sides or bottome of the image
      represented by the pixels array'''
   def checkForBlackEdges(self, pixels, width, height):
      blackEdge = False

      #Check for black edge along bottom
      for x in range(width):
         if pixels[x, height - 1] < 128:
            blackEdge = True
         else:
            blackEdge = False
            break

      #There is a black border along the bottom of the image
      if blackEdge:
         return True

      #Check for black border on the left side of the image
      for y in range(height):
         if pixels[0, y] < 128:
            blackEdge = True
         else:
            blackEdge = False
            break

      #There is a black border along the left side of the image
      if blackEdge:
         return True

      #Check for black border on the right side of the image
      for y in range(height):
         if pixels[width - 1, y] < 128:
            blackEdge = True
         else:
            blackEdge = False
            break

      return blackEdge

   '''Split the newly downloaded images into training and validation directories'''
   def splitImages(self, validPercent):
      dirPath = os.getcwd() #Get the current directory path

      #Remove any existing training and validation directories and remake them
      try:
         if os.path.isdir(dirPath + '/Training_Images'):
            shutil.rmtree(dirPath + '/Training_Images')
         if os.path.isdir(dirPath + '/Validation_Images'):
            shutil.rmtree(dirPath + '/Validation_Images')
         os.mkdir(dirPath + '/Training_Images')
         os.mkdir(dirPath + '/Validation_Images')
      except OSError as err:
         print("Error: {0}".format(err))
         return

      #List all the images that have been downloaded, now and previously
      images = os.listdir(dirPath + "/Input_Images")

      #Determine how many images to use for validation
      numValid = round(len(images) * validPercent)
      numChosen = 0;

      #Save images to the validation directory randomly
      while numChosen <= numValid-1:
         index = randint(0, len(images)-1)
         imgName = images.pop(index)
         img = Image.open(dirPath + "/Input_Images/" + imgName)
         img.save(dirPath + "/Validation_Images/" + imgName)
         numChosen += 1;

      #Save the rest of the images to the training directory
      for imgName in images:
         img = Image.open(dirPath + "/Input_Images/" + imgName)
         img.save(dirPath + "/Training_Images/" + imgName)

      return

if __name__ == '__main__':
   d = DataExtractor()
   d._main()
