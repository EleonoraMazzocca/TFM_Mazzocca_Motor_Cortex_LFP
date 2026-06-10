#%%
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import RMSprop # type: ignore
from keras import backend as K
from tqdm.keras import TqdmCallback
import numpy as np
from classification_tasks import ClassificationTask
from sklearn.model_selection import train_test_split
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def create_model_TC2DNN():

    model = tf.keras.models.Sequential([ 
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(256, 500, 1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten()
    ])
    return model

def create_model_TC1DNN():

    model = tf.keras.models.Sequential([ 
            tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(256, 500, 1)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu')
        ])
    return model

class NeuroNN():
    def __init__(self):
        self.adp = None
        self.model = None
        self.scores = None
        self.NLAYERS = 64
    
    @staticmethod
    def simplified_tasks():
        classtasks = ClassificationTask()
        
        tasks = [
            (classtasks.get_task_power_precision        , {"phase" : 1},                                   "power_precision"),

            (classtasks.get_task_unimanual_bimanual     , {"phase" : 1},                                   "unimanual_bimanual"),

            (classtasks.get_task_power_precision_hand   , {"hand" : "L", "phase" : 1},                     "power_precision_L"),
            (classtasks.get_task_power_precision_hand   , {"hand" : "R", "phase" : 1},                     "power_precision_R"),
            (classtasks.get_task_power_precision_nobi   , {"phase" : 1},                                   "power_precision_nobi"),

            (classtasks.get_task_left_right             , {"phase" : 1},                                   "left_right"),
            
            (classtasks.get_task_left_right_power       , {"phase" : 1},                                   "left_right_power"),
            (classtasks.get_task_left_right_precision   , {"phase" : 1},                                   "left_right_precision"),

            (classtasks.get_task_angles_hand            , {"hand" : "L", "pp" : "PRECISION", "phase" : 1}, "angles_L_precision_1"),
            (classtasks.get_task_angles_hand            , {"hand" : "L", "pp" : "POWER", "phase" : 1},     "angles_L_power_1"),
            (classtasks.get_task_angles_hand            , {"hand" : "L", "pp" : "PRECISION", "phase" : 2}, "angles_L_precision_2"),
            (classtasks.get_task_angles_hand            , {"hand" : "L", "pp" : "POWER", "phase" : 2},     "angles_L_power_2"),
            
            (classtasks.get_task_angles_hand            , {"hand" : "R", "pp" : "PRECISION", "phase" : 1}, "angles_R_precision_1"),
            (classtasks.get_task_angles_hand            , {"hand" : "R", "pp" : "POWER", "phase" : 1},     "angles_R_power_1"),
            (classtasks.get_task_angles_hand            , {"hand" : "R", "pp" : "PRECISION", "phase" : 2}, "angles_R_precision_2"),
            (classtasks.get_task_angles_hand            , {"hand" : "R", "pp" : "POWER", "phase" : 2},     "angles_R_power_2"),

            (classtasks.get_task_angles_bimanual        , {"phase" : 2},                                   "angles_bimanual")
        ]

        return classtasks, tasks
    
    def train_model_TC1DNN(self, tasks, data_loader):

        self.adp = create_model_TC1DNN()

        self.scores = np.zeros((len(tasks), len(tasks))) # iteration x cls
        for n_rep in range(len(tasks)):
            for n_model, task in enumerate(tasks):
                # data loading from tasks
                task[0](**task[1])

                # Swap last layers
                current_layers = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(len(data_loader.classes.keys()), activation='softmax')
                ], name=task[2])
                
                # Compile current model with new last layers
                self.model = tf.keras.models.Sequential([
                    self.adp,
                    current_layers
                ], name="n_model_"+str(n_model))

                # Freeze/Unfreeze the model
                if n_model == n_rep:
                    self.adp.trainable = True
                else:
                    self.adp.trainable = False


                self.model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(learning_rate=1e-4),
                    metrics=['accuracy'])
                self.model.summary()

                # Training data
                X = data_loader.X.reshape((data_loader.X.shape[0], data_loader.X.shape[1], data_loader.X.shape[2], 1))
                y = np.zeros((data_loader.y.shape[0], len(data_loader.classes.keys())))
                for i, label in enumerate(data_loader.y):
                    y[i, label] = 1
                X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.25, random_state=42)
                
                # Callbacks
                cbs = [
                    EarlyStopping(
                        monitor='val_accuracy', 
                        patience=10, 
                        verbose=0,
                        restore_best_weights=True,
                        mode='max'),
                    ModelCheckpoint(
                        '.mdl_wts.hdf5', 
                        save_best_only=True, 
                        monitor='val_accuracy', 
                        mode='max'),
                    ReduceLROnPlateau(
                        monitor='val_accuracy', 
                        factor=0.1, 
                        patience=7, 
                        verbose=1, 
                        min_delta=1e-4, 
                        mode='max')
                ]
                
                split = 5
                split_i = X_train.shape[0]//split
                # Fit the model
                for _ in range(2):
                    for i in range(split):
                        start_i = i*split_i
                        stop_i = start_i+split_i
                        if stop_i > X_train.shape[0]: stop_i = -1
                        _ = self.model.fit(
                            X_train[start_i:stop_i],
                            y_train[start_i:stop_i],
                            steps_per_epoch=8,  
                            epochs=5,
                            verbose=2,
                            validation_data=(X_val, y_val),
                            callbacks=cbs
                        )
                
                # Test the model
                predictions = self.model.predict(X_val)
                pred = np.argmax(predictions, axis=1)
                y_val = np.argmax(y_val, axis=1)
                c = 0
                for i in range(y_val.shape[0]):
                    if y_val[i] == pred[i]:
                        c += 1

                # Save the models
                self.adp.save("./models/adp_model.h5")
                current_layers.save("./models/"+task[2]+".h5")

                # Clear GPU Memory
                K.clear_session()
                gc.collect()
                del self.model
                del current_layers
                del self.adp

                self.adp = create_model_TC1DNN()
                self.adp.build()
                self.adp.load_weights("./models/adp_model.h5")

                print("-="*100)
                print(task[2])
                print(c/pred.shape[0])
                print("-="*100)
                self.scores[n_rep, n_model] = c/pred.shape[0]
        print(self.scores)
    
    # first train
    def train_model_TC2DNN(self, tasks, data_loader):
        self.adp = create_model_TC2DNN()
        self.scores = np.zeros((len(tasks), len(tasks))) # iteration x cls
        for n_rep in range(len(tasks)):
            for n_model, task in enumerate(tasks):
                # data loading from tasks
                task[0](**task[1])

                # Swap last layers
                current_layers = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(self.NLAYERS, activation='relu'),
                    tf.keras.layers.Dense(len(data_loader.classes.keys()), activation='softmax')
                ], name=task[2])
                
                # Compile current model with new last layers
                self.model = tf.keras.models.Sequential([
                    self.adp,
                    current_layers
                ], name="n_model_"+str(n_model))

                # Freeze/Unfreeze the model
                if n_model == n_rep:
                    self.adp.trainable = True
                else:
                    self.adp.trainable = False


                self.model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(learning_rate=1e-4),
                    metrics=['accuracy'])
                self.model.summary()

                # Training data
                X = data_loader.X.reshape((data_loader.X.shape[0], data_loader.X.shape[1], data_loader.X.shape[2], 1))
                y = np.zeros((data_loader.y.shape[0], len(data_loader.classes.keys())))
                for i, label in enumerate(data_loader.y):
                    y[i, label] = 1
                X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.25, random_state=42)
                
                # Callbacks
                cbs = [
                    EarlyStopping(
                        monitor='val_accuracy', 
                        patience=10, 
                        verbose=0,
                        restore_best_weights=True,
                        mode='max'),
                    ModelCheckpoint(
                        '.mdl_wts.hdf5', 
                        save_best_only=True, 
                        monitor='val_accuracy', 
                        mode='max'),
                    ReduceLROnPlateau(
                        monitor='val_accuracy', 
                        factor=0.1, 
                        patience=7, 
                        verbose=1, 
                        min_delta=1e-4, 
                        mode='max')
                ]
                
                split = 5
                split_i = X_train.shape[0]//split
                # Fit the model
                for _ in range(2):
                    for i in range(split):
                        start_i = i*split_i
                        stop_i = start_i+split_i
                        if stop_i > X_train.shape[0]: stop_i = -1
                        _ = self.model.fit(
                            X_train[start_i:stop_i],
                            y_train[start_i:stop_i],
                            steps_per_epoch=8,  
                            epochs=5,
                            verbose=2,
                            validation_data=(X_val, y_val),
                            callbacks=cbs
                        )
                
                # Test the model
                predictions = self.model.predict(X_val)
                y_val = np.argmax(y_val, axis=1)
                pred = np.argmax(predictions, axis=1)
                c = 0
                for i in range(y_val.shape[0]):
                    if y_val[i] == pred[i]:
                        c += 1

                # Save the models
                self.adp.save("./models/adp_model.h5")
                current_layers.save("./models/"+task[2]+".h5")

                # Clear GPU Memory
                K.clear_session()
                gc.collect()
                del self.model
                del current_layers
                del self.adp

                self.adp = create_model_TC2DNN()
                self.adp.build()
                self.adp.load_weights("./models/adp_model.h5")

                print("-="*100)
                print(task[2])
                print(c/pred.shape[0])
                print("-="*100)
                self.scores[n_rep, n_model] = c/pred.shape[0]
        print(self.scores)
    
    def test_model(self, tasks, data_loader):
        self.adp = create_model_TC1DNN()
        for n_model, task in enumerate(tasks):
            # data loading from tasks
            task[0](**task[1])

            # Swap last layers
            current_layers = tf.keras.models.Sequential([
                #tf.keras.layers.Dense(self.NLAYERS, activation='relu'),
                tf.keras.layers.Dense(len(data_loader.classes.keys()), activation='softmax')
            ], name=task[2])
            
            # Compile current model with new last layers
            self.model = tf.keras.models.Sequential([
                self.adp,
                current_layers
            ], name="n_model_"+str(n_model))
            self.model.build()
            self.adp.load_weights("./models/adp_model.h5")
            current_layers.load_weights("./models/"+task[2]+".h5")

            # Test data
            X = data_loader.X.reshape((data_loader.X.shape[0], data_loader.X.shape[1], data_loader.X.shape[2], 1))
            y = np.zeros((data_loader.y.shape[0], len(data_loader.classes.keys())))
            for i, label in enumerate(data_loader.y):
                y[i, label] = 1
            _, X_val, _, y_val = train_test_split(X,y, test_size=0.25, random_state=42)

            predictions = self.model.predict(X_val)
            pred = np.argmax(predictions, axis=1)
            y_val = np.argmax(y_val, axis=1)
            c = 0
            for i in range(y_val.shape[0]):
                if y_val[i] == pred[i]:
                    c += 1

            K.clear_session()
            gc.collect()
            del self.model
            del current_layers

            print("-="*100)
            print(task[0].__name__)
            print(c/pred.shape[0])
            print("-="*100)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

neuro_nn = NeuroNN()

simp_cl, simp_tasks = NeuroNN.simplified_tasks()

neuro_nn.train_model_TC2DNN(simp_tasks, simp_cl)
#neuro_nn.test_model(simp_tasks, simp_cl)
print(neuro_nn.scores)
from datetime import datetime
np.save("scores_nn_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), neuro_nn.scores)


