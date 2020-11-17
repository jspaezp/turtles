

# Saving the turtles by using AI

This is supposed to contain the "library code" for my submission for
https://zindi.africa/competitions/local-ocean-conservation-sea-turtle-face-detection

Please check the competition if you are interested

I am intending make this entry using Detectron2 (because why not ...). It makes training a lot faster and easier. I intend to use the Retinanet model as a base for the classifier.

Components

"Library components"
-  visualization.py
   -  Contains the functionality to translate the output of the model to human-interpretable formats (annotated images).
   -  As well as the functionality to check visually the successful generation Detectron2 dataset objects.
-  augmentation.py **NOT IMPLEMENTED**
   -  TODO: contains the functionality to augment the images
-  t_io.py
   - Contains the functions to read the data, both as images and as Detectron2 datasets.
   - Also registers the datasets to Detectron
   - Also contains some of the functions that handle the conversion to and from the submission formats.


"User components"

-  data_spltting.py
   -  contains the functions to split the data into training and testing sets, and arranges it in sub-directories (simlinks them, more accurately ...). It also creates a csv with the new arranged information for each directory.
   -  usage: `python data_splitting.py --help`
   -  `python data_splitting.py --in_file ../turtles_data/Train.csv --img_dir ../turtles_data/IMAGES_512 --out_dir ../turtles_data/split --fraction 0.8`
-  prediction.py
   -  Contains functions that get an image, applies any transformation require and outputs the detected regions and an image with the annotation on it, as well as internal functions to deal with the predictions.
   -  Usage: `python prediction.py` After training the model
   -  `python prediction.py --in_img my_image.JPG --out_img my_predictions.PNG`
-  train.py
   -  Contains all the train instructions and runnable in the command line
   -  NOT USABLE RIGHT NOW: `python train.py --split_dir ./split --out_dir ./out`

## Development

- use `black` to format your code :)

## Dependencies

The installation of detectron and pytorch is a lil bit complicated but the rest are available though pip

- click
- opencv (cv2)
- pandas
- numpy
- detectron2