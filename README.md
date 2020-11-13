

# Saving the turtles by using AI

This is supposed to contain the "library code" for my submission for
https://zindi.africa/competitions/local-ocean-conservation-sea-turtle-face-detection

Please check the competition if you are interested

I am intending make this entry using Detectron2 (because why not ...)

Components

"Library components"
-  visualization.py
   -  TODO: contains the functionality to translate the output of the model to human-interpretable formats
-  augmentation.py
   -  TODO: contains the functionality to augment the images
-  io.py
   -  TODO: contains the functions to read the data, both as images and as detectron datasets


"User components"

-  data_spltting.py
   -  contains the functions to split the data into training and testing sets, and arranges it in sub-directories (simlinks them, more accurately ...)
   -  usage: python data_splitting.py --help
   -  `python data_splitting.py --in_file ../turtles_data/Train.csv --img_dir ../turtles_data/IMAGES_512 --out_dir ../turtles_data/split --fraction 0.8`
-  prediction.py
   -  TODO: should contain a function that gets an image, applies any transformation require and outputs the detected regions and an image with the annotation on it
-  train.py
   -  Contains all the train instructions and runnable in the command line
   -  `python train.py --split_dir ./split --out_dir ./out`

## Development

- use `black` to format your code :)