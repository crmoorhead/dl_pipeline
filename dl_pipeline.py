# ALL FUNCTION USED IN DEEP LEARNING PIPELINE

import os
from random import shuffle, seed as sd, sample, choice
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, 'C:\\Users\\the_n\\OneDrive\\Python Programs\\Sonar Project')
from network_structure import *
from classifiers import *
from pretrained import *
from model_analysis import *
from network_structure import *
from image_toolbox import *

def classification_dict(source,*args):
    dictionary = {}
    input_type=source.__class__
    # Create a dictionary comprising of lists of files of each class contained in a given path
    if input_type==str:
        file_names=os.listdir(source)
        for f in file_names:
            if "ignore_subclasses" in args:
                label=f.split("_")[0]
            else:
                label=f.split(" ")[0]
            if label in dictionary:
                dictionary[label].append(f)
            else:
                dictionary[label]=[f]
        if "class_stats" in args:
            class_stats = {c: len(dictionary[c]) for c in dictionary}
    # Create a dictionary of items that can used to create an input to a CNN arranged by class label
    elif input_type in [dict,list]:
        if "class_stats" in args:
            class_stats = {}
        if "balance_subclasses" not in args:
            for d in source:
                if input_type==dict:
                    one_label=list(source[d]).index(1)
                    if one_label not in dictionary:
                        dictionary[one_label] = {d: source[d]}
                    else:
                        dictionary[one_label][d] = source[d]
                    if "class_stats" in args:
                        class_stats = {c: len(dictionary[c]) for c in dictionary}
                else:
                    one_label=d[1].index(1)
                    if one_label not in dictionary:
                        dictionary[one_label]={d[0]:d[1]}
                    else:
                        dictionary[one_label][d[0]]=d[1]
                    if "class_stats" in args:
                        if one_label not in class_stats:
                            class_stats[one_label] = 1
                        else:
                            class_stats[one_label] += 1
        else:
            if "class_stats" in args:
                subclass_stats={}
            for d in source:
                if input_type==dict:
                    one_label=list(source[d]).index(1)
                else:
                    one_label = d[1].index(1)
                    if "class_stats" in args:
                        if one_label not in class_stats:
                            class_stats[one_label] = 1
                        else:
                            class_stats[one_label] += 1
                if one_label not in dictionary:
                    dictionary[one_label]={}
                if input_type==dict:
                    subclass=d.split("_")[1].split(" ")[0]
                    if subclass not in dictionary[one_label]:
                        dictionary[one_label][subclass] = {d: source[d]}
                    else:
                        dictionary[one_label][subclass][d] = source[d]
                    subclass_stats={c:{len(dictionary[c][s]) for s in dictionary[c]} for c in dictionary}
                else:
                    subclass=d[0].split("_")[1].split(" ")[0]
                    if subclass not in dictionary[one_label]:
                        dictionary[one_label][subclass]={d[0]:d[1]}
                    else:
                        dictionary[one_label][subclass][d[0]]=d[1]
                    if "class_stats" in args:
                        if one_label not in subclass_stats:
                            subclass_stats[one_label] = {}
                        if subclass not in subclass_stats[one_label]:
                            subclass_stats[one_label][subclass] = 1
                        else:
                            subclass_stats[one_label][subclass] += 1

    else:
        pass

    if "class_stats" in args:
        if "balance_subclasses" not in args:
            print("Number of Instances per Class Label")
            print(class_stats)
        else:
            for c in subclass_stats:
                print("Class:",c,"Total:",class_stats[c])
                print(subclass_stats[c])

    return dictionary

def train_val_test(dictionary, ratios, seed):
    order={}
    class_sizes={}
    for d in dictionary:
        order[d]=list(range(len(dictionary[d])))
        if seed!=None:
            sd(seed)
        shuffle(order[d])
        class_sizes[d]=len(order[d])

    splits=[sum(ratios[:i+1])/sum(ratios) for i in range(2)]

    train_indices={c:order[c][:int(splits[0]*class_sizes[c])] for c in class_sizes}
    val_indices={c:order[c][int(splits[0]*class_sizes[c]):int(splits[1]*class_sizes[c])] for c in class_sizes}
    test_indices={c:order[c][int(splits[1]*class_sizes[c]):] for c in class_sizes}

    train_files={c:[dictionary[c][i] for i in train_indices[c]] for c in train_indices}
    val_files={c:[dictionary[c][i] for i in val_indices[c]] for c in train_indices}
    test_files={c:[dictionary[c][i] for i in test_indices[c]] for c in train_indices}

    return train_files,val_files,test_files

def input_output(train_files,val_files,test_files):
    train_inputs,val_inputs,test_inputs={},{},{}
    i=0
    empty_vector=np.array([0]*len(train_files))
    for c in train_files:
        one_hot=empty_vector.copy()
        one_hot[i]=1
        for file in train_files[c]:
            train_inputs[file]=one_hot
        for file in val_files[c]:
            val_inputs[file]=one_hot
        for file in test_files[c]:
            test_inputs[file]=one_hot
        i+=1
    return train_inputs, val_inputs, test_inputs

def show_image(source):
    if source.__class__==str:
        im=cv2.imread(source)
    elif source.__class__==np.ndarray:
        im=source
    else:
        print("Source Not of Correct Type")
        return None
    cv2.imshow(None,im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def batch_sampling(source_dir,subset,batch_size,*args,**kwargs):
    if "balance_classes" not in args and "balance_subclasses" not in args:
        batch_list=[[s,subset[s]] for s in sample(list(subset),batch_size)]
        if "greyscale" in args or "grayscale" in args:
            batch_array=np.expand_dims(np.array([np.array(cv2.imread(source_dir+"\\"+f[0]))[:,:,0] for f in batch_list]),-1)/255
        else:
            batch_array=np.array([np.array(cv2.imread(source_dir+"\\"+f[0])) for f in batch_list])/255
    else:
        new_dict=classification_dict(subset, *args,**kwargs)
        batch_list=[]
        for i in range(batch_size):
            class_choice = choice(list(new_dict.keys()))
            if "balance_subclasses" not in args:
                instance_choice = choice(list(new_dict[class_choice].keys()))
                batch_list.append([instance_choice,new_dict[class_choice][instance_choice]])
            else:
                subclass_choice=choice(list(new_dict[class_choice].keys()))
                instance_choice=choice(list(new_dict[class_choice][subclass_choice].keys()))
                batch_list.append([instance_choice,new_dict[class_choice][subclass_choice][instance_choice]])
        if source_dir==None:
            return batch_list
        if "greyscale" in args or "grayscale" in args:
            batch_array = np.expand_dims(np.array([np.array(cv2.imread(source_dir + "\\" + f[0]))[:, :, 0] for f in batch_list]), -1) / 255
        else:
            batch_array = np.array([np.array(cv2.imread(source_dir + "\\" + f[0])) for f in batch_list]) / 255
    return batch_list, batch_array


def save_model(directory,file_name,model,*args):
    if "weights_only" in args:
        model.save_weights(directory+"\\"+file_name+".h5")
    elif "model_only" in args:
        structure = model.to_json()
        with open(directory+"\\"+file_name+".json", "w") as json_file:
            json_file.write(structure)
    else:
        model.save(directory+"\\"+file_name+".h5")

def load_model(source_dir, file_name,*args,**kwargs):
    if "model_only" in args:
        from keras.models import model_from_json
        load_file = open(source_dir+"\\"+file_name)
        loaded=model_from_json(load_file.read())
        load_file.close()
        if "summary" in args:
            print(loaded.summary())
    elif "weights_to_model" in args:
        if "model" not in kwargs:
            print("No model input to load weights into.")
            return None
        else:
            model=kwargs["model"]
            loaded=model.load_weights(source_dir+"\\"+file_name)
    else:
        from keras.models import load_model
        loaded=load_model(source_dir+"\\"+file_name)
        if "summary" in args:
            print(loaded.summary())
    return loaded

def compiled_model(model,optimiser,loss_func,*args):
    model.compile(optimizer=optimiser, loss=loss_func, metrics=list(args))

def plot_train_val(train_stats,val_stats,*args,**kwargs):
    epochs=[i for i in range(len(train_stats))]
    if "plot_col" in kwargs:
        train_color=kwargs["plot_col"][0]
        val_color = kwargs["plot_col"][1]
    else:
        train_color = "b"
        val_color = "g"
    fig = plt.figure()
    plt.plot(epochs,train_stats,train_color)
    if val_stats is not None:
        plt.plot(epochs,val_stats,val_color)
    else:
        val_stats=train_stats.copy()
    plt.xlabel("EPOCHS")
    plt.ylabel("ACCURACY")
    if "autoplot" in args:
        plt.ylim([min(min(train_stats,val_stats))*0.9, min(max(max(train_stats,val_stats))*1.1,1)])
    if "plot_range" in kwargs:
        plt.ylim(kwargs["plot_range"])
    else:
        plt.ylim(kwargs[0,1])
    if "model_name" in kwargs:
        plt.title(kwargs["model_name"])
    if "save_dir" in kwargs:
        if "model_name" in kwargs:
            fig.savefig(kwargs["save_dir"]+"\\"+kwargs["model_name"]+".png")
        else:
            fig.savefig(kwargs["save_dir"] + "\\untitled_plot.png")

def bitesize(iodict,*args,**kwargs):
    from itertools import islice
    if "bite" in kwargs:
        if kwargs["bite"]=="all":
            chunk_size=len(iodict)
        else:
            chunk_size=int(kwargs["bite"])
    else:
        chunk_size=200
    current=0
    it = iter(iodict)
    chunky_dicks=[{k: iodict[k] for k in islice(it, chunk_size)} for i in range(0, len(iodict), chunk_size)]
    return chunky_dicks

# augment settings dict --> choose augment --> apply augment (to batch or to each image)

# Need valid dict object and augment functional predefined

def batch_train_model(model,source_dir,ioTR,ioV,batch_size,steps,epochs,*args,**kwargs):
    epoch=1
    train_history=[]
    train_chunks=bitesize(ioTR,*args,**kwargs)
    if ioV != {}:
        val_list,val_data=batch_sampling(source_dir,ioV,len(ioV), *args) # We take all of the validation set at once
        best_val=0 # Used if we are saving the best model
    if "timer" in args:
        from timeit import default_timer
        start=default_timer()
    if "save_dir" in kwargs:
        save_model(kwargs["save_dir"],"model_structure",model,"model_only")
    if "save_thresh" in kwargs:
        if kwargs["save_thresh"].__class__!=str:
            if kwargs["save_thresh"] <0 or kwargs["save_thresh"]>1:
                print("Nonesense threshhold given, assigned thresshhold of 0.9 instead.")
                kwargs["save_thresh"]=0.9
    while epoch <=epochs:
        if "timer" in args:
            epoch_start=default_timer()
        print("EPOCH ",epoch)
        epoch_history=[]
        step=1
        # PROCESS BATCHES FOR CURRENT EPOCH
        while step<=steps:
            current_list,current_batch=batch_sampling(source_dir,ioTR,batch_size, *args)
            if "augmentations" in kwargs:
                pass
            out_array=np.array([current_list[i][1] for i in range(len(current_list))])
            batch_stats=model.train_on_batch(current_batch,out_array)
            epoch_history.append(batch_stats)
            step+=1

        # ACTIONS TO DO AT END OF EPOCH
        if "timer" in args:
            print("Epoch duration:", str((default_timer()-epoch_start)/60)[:4],"minutes")
        print("Evaluating Accuracy on Training Set")
        train_scores=[]
        train_start=default_timer()
        for t in train_chunks:
            train_batch_list, train_batch_data = batch_sampling(source_dir, t, len(t), *args)
            train_scores.append(model.evaluate(train_batch_data, np.array([train_batch_list[i][1] for i in range(len(train_batch_list))]) , verbose=1)[1]*len(t))
        train_score=sum(train_scores)/len(ioTR) # why a list in a list?
        print("Training score:",train_score)
        if "timer" in args:
            print("Training Score Time:", (default_timer()-train_start)/60,"mins")
        if ioV !={}:
            print("Evaluating Accuracy on Validation Set")
            val_score = model.evaluate(val_data, np.array([val_list[i][1] for i in range(len(val_list))]) , verbose=1)
            print("Accuracy on Validation Set:", str(val_score[1])[:6])
            if "save_dir" in kwargs:
                if "save_thresh" in kwargs:
                    if kwargs["save_thresh"]=="best":
                        if val_score[1]>=best_val:
                            save_model(kwargs["save_dir"],"EPOCH_"+str(epoch)+" (val_acc="+str(val_score[1])[:6]+")",model,"weights_only")
                            best_val=val_score[1]
                else:
                    if val_score[1]>=kwargs["save_thresh"]:
                        save_model(kwargs["save_dir"],"EPOCH_"+str(epoch)+" (val_acc="+str(val_score[1])[:6]+")",model,"weights_only")
            train_history.append([train_score,val_score])
        else:
            train_history.append(train_score)
        epoch+=1
        print()

    # ACTIONS AT END OF TRAINING
    if "save_dir" in kwargs and "save_final" in args:
        save_model(kwargs["save_dir"], "FINAL MODEL (val_acc=" + str(val_score[1])[:6] + ")", model, "weights_only")

    if "save_dir" not in kwargs:
        return model

    if "timer" in args:
        print("Training duration:",str((default_timer()-start)//60)[:4],"minutes")

    if "save_plot" in args:
        if ioV !={}:
            plot_train_val(train_history[0],train_history[1],*args,**kwargs)
        else:
            pass