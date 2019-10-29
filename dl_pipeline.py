# ALL FUNCTION USED IN DEEP LEARNING PIPELINE

import os
from random import shuffle, seed as sd, sample
import numpy as np
import cv2
import matplotlib.pyplot as plt

def classification_dict(source_dir,*args):
    file_names=os.listdir(source_dir)
    dictionary={}
    for f in file_names:
        if "subclasses" in args:
            label=f.split("_")[0]
        else:
            label=f.split(" ")[0]
        if label in dictionary:
            dictionary[label].append(f)
        else:
            dictionary[label]=[f]
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

def batch_sampling(source_dir,subset,batch_size,*args):
    batch_dict={s:subset[s] for s in sample(list(subset),batch_size)}
    if "greyscale" in args:
        batch_array=np.expand_dims(np.array([np.array(cv2.imread(source_dir+"\\"+f))[:,:,0] for f in batch_dict]),-1)/255
    else:
        batch_array=np.array([np.array(cv2.imread(source_dir+"\\"+f)) for f in batch_dict])/255
    return batch_dict, batch_array


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
    fig = plt.figure()
    plt.plot(epochs,train_stats,'b')
    plt.plot(epochs,val_stats,'g')
    plt.xlabel("EPOCHS")
    plt.ylabel("ACCURACY")
    if "model_name" in kwargs:
        plt.title(kwargs["model_name"])
    if "save_dir" in kwargs:
        if "model_name" in kwargs:
            fig.savefig(kwargs["save_dir"]+"\\"+kwargs["model_name"]+".png")

# TEST
save_dir="C:\\Users\\the_n\\Documents\\PhD Stuff\\Paper Writing\\First Paper"
train=[0.5,0.6,0.7,0.8]
val=[0.7,0.7,0.7,0.8]


plot_train_val(train,val,"show",model_name="MODEL A",save_dir=save_dir)

def batch_train_model(model,source_dir,ioTR,ioV,batch_size,steps,epochs,*args,**kwargs):
    epoch=1
    train_history=[]
    if "val_acc" in args:
        val_list,val_data=batch_sampling(source_dir,ioV,len(ioV), *args)
        best_val=0
    if "timer" in args:
        from timeit import default_timer
        start=default_timer()
    if "save_best" in kwargs:
        save_model(kwargs["save_best"],"model_structure",model,"model_only")
    while epoch <=epochs:
        if "timer" in args:
            epoch_start=default_timer()
        print("EPOCH ",epoch)
        epoch_history=[]
        step=1
        while step<=steps:
            current_list,current_batch=batch_sampling(source_dir,ioTR,batch_size, *args)
            if "augmentations" in kwargs:
                pass
            # AUGMENTATION PHASE TO DO
            out_array=np.array(list(current_list.values()))
            batch_stats=model.train_on_batch(current_batch,out_array)
            epoch_history.append(batch_stats)
            step+=1
        # ACTIONS TO DO AT END OF EPOCH
        if "timer" in args:
            print("Epoch duration:", str((default_timer()-epoch_start)/60)[:4],"minutes")
        if "val_acc" in args:
            print("Testing on Validation Set")
            val_score = model.evaluate(val_data, np.array(list(val_list.values())) , verbose=1)
            print("Accuracy on Validation Set:", str(val_score[1])[:6])
            if "save_best" in kwargs:
                if val_score[1]>=best_val:
                    save_model(kwargs["save_best"],"EPOCH_"+str(epoch)+" (val_acc="+str(val_score[1])[:6]+")",model,"weights_only")
        epoch+=1

    save_model(kwargs["save_best"],"FINAL MODEL (val_acc="+str(val_score[1])[:6]+")",model,"weights_only")

    #print(epoch_history)
    #print(train_history)

    # STUFF TO DO AT END OF TRAINING
        # MAKE PLOT
    if "timer" in args:
        print("Training duration:",str((default_timer()-start)//60)[:4],"minutes")

