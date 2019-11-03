# dl_pipeline

The Deep Learning pipeline for image classification uses and adds to the function already written in the other repositories (namely image_toolbox, augmentation, network_structure and model analysis). The new functions added are split into how they are applied chronologically in the pipeline

## Data Preparation

These functions are used to prepare the files to have suitable input and labelled output for training a model.

- **classification_dict(source_dir,\*args)**: This takes in one argument, the path of a directory containing all images we have at our disposal. The images must be named according to a given convention, that is all of the form _class (#)_ or _class_subclass (#)_. For the first form, the image label and instance number are given. For the second, the image label can be annotated with subclasses where each subclass is populated by instances of the same or similar objects which are examples of the greater class. The distinction is so that we can do a more detailed comparison at a later date. The output of this fuction is a dictionary with keywords corresponding to classes and values corresponding to lists of the filenames of all instances of that class.

- **train_val_test(dictionary, ratios, seed)**: This takes in three arguements. The dictionary generated bu the classification_dict above, the ratios of the relative sizes of the training, validation and test sets and the random seed used to make this selection. The ratios argument is a list of positive integers, although it is possible to set some of these to zero if so required. The random seed can be set to None, but the reason we require input is because the function requires a random shuffle before splitting into subsets and we usually wish to perform repeatable experiments on the same test and training set splits. ==The output will be three dictionaries which have keys pertaining to the filename and values of the class of the object.

- **input_output(train_files,val_files,test_files)**: This takes in the output dictionaries from the train_val_test function and changes the labels to the one-hot vectors needed for training. 

- **show_image(source)**: This takes in an image in the form of a path string or a numpy array and displays it to the user. This is not a standard function in OpenCV or PIL so it is useful to have a simple single-line way to do this.

These functions can be combined in one line to provide a dictionary of all data available split into training, validation and test sets with one-hot labels given a single directory location. 

## Training Functions

The following funtions are required to perform training and enable extra functionality.

- **batch_sampling(source_dir,subset,batch_size,\*args)**: This function selects a batch of images to be loaded to the workspace in the format of a numpy array that can be used for training a CNN. By default, these arrays will have three channels as that is how jpegs are interpreted by OpenCV, so if we are dealing with greyscale, then we need to use the optional argument "greyscale" (alternatively "grayscale" to import only one channel arrays. The source_dir is where we find the filenames in the subset we are drawing the samples from. This subset is usually the dictionary associated with the training set output by the input_output function above. The batch size is an integer and is the number of samples in each batch. It gives a pair of outputs - a batch dictionary and a batch array. The batch dictionary is of the same form as the subset input whereas the batch array is a single 4D array. We require two outputs as we want a single array object to pass into training whereas the dictionary contains label information for each instance.

- **save_model(directory,file\_name,model,\*args)**: This function saves a model to file to be used later. It must be given an intended directory and file name. With optional arguments "weights_only" and "model_only", it saves only the weights (to be loaded to the same original structure) or only the model (to be used completely from scratch at another time). This function is used internally during the training process to save as training improves.

- **load_model(source_dir,file\_name,\*args,\*\*kwargs)**: The compliment of the save_model function where we can load a model that has been previously constructed and/or trained. By default, it loads a model with weights and returns a keras model object. The optional arguments "models_only" and "weights_to_model" allow the empty model or the weights to be imprinted to an existing model of the correct architecture. The latter requires a compulsory model keyword with value being the model object. If the "summary" argument is given, the structure of the model is printed for the user.

- **compiled_model(model, optimiser, loss_func,\*args)**: A trivial helper function that compiles the model. All models, included loaded models, must be compiled to be used in training or prediction. The model argument is a keras model object and the optimiser and loss_func are valid keras optimiser and loss function objects. The metrics are defined be adding additional string arguments that correspond to existing keras metrics. 

- **plot_train_val(train_stats,val_stats,\*args,\*\*kwargs)**: This plots the training and validation accuracies during the training process. It is a necessary ingredient of the batch_train_model function as the user may want to review this plot when training is completed. The plot is saved to the same folder as the models, specified by the "save_dir" keyword argument. It will plot the epochs on the x-axis and the accuracy on the y-axis. The limits of the y-axis are by default the interval \[0,1\], but if the model trains well it is hard to see the difference between accurate models and so a finer grade might be more useful. Fixing the range can be done by entering the "plot_range" keyword with value being a tuple of the upper and lower limit of the plot. Alternatively, the optional "autoplot" argument with adjust the y-axis to fit the data. If the keyword "model_name" is given, the plot is given a title with the name of the model. The defualt plot colours are blue for training and green for validation, but these can be changed by passing a tuple of valid colour definitions (see matplotlib documentation) as a tuple to the keyword "plot_col". If there is only training data and not validation data, None must be entered for the val_stats argument.

- **batch_train_model(model,source_dir,ioTR,ioV, batch_size, steps, epochs,\*args,\*\*kwargs)**: This function trains the model by batches of the specified size. The compiled model object must be the first argument, followed by the path of the source directory of the images, the dictionaries pertaining to the Training and Validation sets as generated by input_output, the size of each batch, the number of batches in each epoch and the numbr of epochs. If the validation set is an empty dictionary then the validation step is automatically skipped. If the "timer" argument is given, then the time








