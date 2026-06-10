#%%
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from keras import backend as K
from tqdm.keras import TqdmCallback
import numpy as np
from baseline_linear_classifier.classification_tasks import ClassificationTask
from sklearn.model_selection import train_test_split
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def validate_class_data_paths():
    """
    Ensure all expected class .npy files exist before starting a long training run.
    """
    loader = ClassificationTask()
    missing = [p for p in loader.locations.values() if not os.path.exists(p)]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} class data files. First paths:\n{preview}"
        )

def create_model():

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

class NeuroNN():
    def __init__(self):
        self.adp = create_model()
        self.model = None
        self.scores = None
        """tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        """
    
    @staticmethod
    def multioutput_task():
        classtasks = ClassificationTask()
        
        # Current classification_tasks.py exposes get_task_multioutput_test.
        task = classtasks.get_task_multioutput_test

        return classtasks, task


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


    @staticmethod
    def CC_test():
        classtasks = ClassificationTask()
        
        tasks = [
            (classtasks.get_task_power_precision        , {"phase" : 1},                                   "power_precision",       4),

            (classtasks.get_task_unimanual_bimanual     , {"phase" : 1},                                   "unimanual_bimanual",    0),

            (classtasks.get_task_power_precision_hand   , {"hand" : "L", "phase" : 1},                     "power_precision_L",     4),
            (classtasks.get_task_power_precision_hand   , {"hand" : "R", "phase" : 1},                     "power_precision_R",     4),
            (classtasks.get_task_power_precision_nobi   , {"phase" : 1},                                   "power_precision_nobi",  4),

            (classtasks.get_task_left_right             , {"phase" : 1},                                   "left_right",            1),
            
            (classtasks.get_task_left_right_power       , {"phase" : 1},                                   "left_right_power",      1),
            (classtasks.get_task_left_right_precision   , {"phase" : 1},                                   "left_right_precision",  1),

            (classtasks.get_task_angles_hand            , {"hand" : "L", "pp" : "PRECISION", "phase" : 1}, "angles_L_precision_1",  2),
            (classtasks.get_task_angles_hand            , {"hand" : "L", "pp" : "POWER", "phase" : 1},     "angles_L_power_1",      2),
            (classtasks.get_task_angles_hand            , {"hand" : "L", "pp" : "PRECISION", "phase" : 2}, "angles_L_precision_2",  2),
            (classtasks.get_task_angles_hand            , {"hand" : "L", "pp" : "POWER", "phase" : 2},     "angles_L_power_2",      2),
            
            (classtasks.get_task_angles_hand            , {"hand" : "R", "pp" : "PRECISION", "phase" : 1}, "angles_R_precision_1",  3),
            (classtasks.get_task_angles_hand            , {"hand" : "R", "pp" : "POWER", "phase" : 1},     "angles_R_power_1",      3),
            (classtasks.get_task_angles_hand            , {"hand" : "R", "pp" : "PRECISION", "phase" : 2}, "angles_R_precision_2",  3),
            (classtasks.get_task_angles_hand            , {"hand" : "R", "pp" : "POWER", "phase" : 2},     "angles_R_power_2",      3)
        ]

        return classtasks, tasks
    
    @staticmethod
    def CC1D_test():
        classtasks = ClassificationTask()
        
        tasks = [
            (classtasks.get_task_power_precision        , {"phase" : 1},                                   "power_precision",       [12,13]),

            (classtasks.get_task_unimanual_bimanual     , {"phase" : 1},                                   "unimanual_bimanual",    [0,1]),

            (classtasks.get_task_power_precision_hand   , {"hand" : "L", "phase" : 1},                     "power_precision_L",     [12,13]),
            (classtasks.get_task_power_precision_hand   , {"hand" : "R", "phase" : 1},                     "power_precision_R",     [12,13]),
            (classtasks.get_task_power_precision_nobi   , {"phase" : 1},                                   "power_precision_nobi",  [12,13]),

            (classtasks.get_task_left_right             , {"phase" : 1},                                   "left_right",            [2,3]),
            
            (classtasks.get_task_left_right_power       , {"phase" : 1},                                   "left_right_power",      [2,3]),
            (classtasks.get_task_left_right_precision   , {"phase" : 1},                                   "left_right_precision",  [2,3]),

            (classtasks.get_task_angles_hand            , {"hand" : "L", "pp" : "PRECISION", "phase" : 1}, "angles_L_precision_1",  [4,5,6,7]),
            (classtasks.get_task_angles_hand            , {"hand" : "L", "pp" : "POWER", "phase" : 1},     "angles_L_power_1",      [4,5,6,7]),
            (classtasks.get_task_angles_hand            , {"hand" : "L", "pp" : "PRECISION", "phase" : 2}, "angles_L_precision_2",  [4,5,6,7]),
            (classtasks.get_task_angles_hand            , {"hand" : "L", "pp" : "POWER", "phase" : 2},     "angles_L_power_2",      [4,5,6,7]),
            
            (classtasks.get_task_angles_hand            , {"hand" : "R", "pp" : "PRECISION", "phase" : 1}, "angles_R_precision_1",  [8,9,10,11]),
            (classtasks.get_task_angles_hand            , {"hand" : "R", "pp" : "POWER", "phase" : 1},     "angles_R_power_1",      [8,9,10,11]),
            (classtasks.get_task_angles_hand            , {"hand" : "R", "pp" : "PRECISION", "phase" : 2}, "angles_R_precision_2",  [8,9,10,11]),
            (classtasks.get_task_angles_hand            , {"hand" : "R", "pp" : "POWER", "phase" : 2},     "angles_R_power_2",      [8,9,10,11])
        ]

        return classtasks, tasks

    def train_model_CCBNN(self, task, data_loader):
        N_NEURONS = 256
        input_layer = tf.keras.layers.Input(shape=(256, 500, 1))

        # Adaptive convolutional layers
        conv1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(input_layer)
        mp1 = tf.keras.layers.MaxPooling2D(2,2)(conv1)
        conv2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(mp1)
        mp2 = tf.keras.layers.MaxPooling2D(2,2)(conv2)
        conv3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(mp2)
        mp3 = tf.keras.layers.MaxPooling2D(2,2)(conv3)
        flatten = tf.keras.layers.Flatten()(mp3)
        last_out = tf.keras.layers.Dense(1024, activation='relu', name="last_out")(flatten)

        # Unimanual/Bimanual
        uni_bi_dense = tf.keras.layers.Dense(N_NEURONS, activation='relu', name="uni_bi_dense")(last_out)
        uni_bi_out = tf.keras.layers.Dense(2, activation='softmax', name="uni_bi_out")(uni_bi_dense)

        # Left/Right
        left_right_dense = tf.keras.layers.Dense(N_NEURONS, activation='relu', name="left_right_dense")(last_out)
        left_right_out = tf.keras.layers.Dense(2, activation='softmax', name="left_right_out")(left_right_dense)

        # Left angles
        left_angles_dense = tf.keras.layers.Dense(N_NEURONS, activation='relu', name="left_angles_dense")(last_out)
        left_angles_out = tf.keras.layers.Dense(4, activation='softmax', name="left_angles_out")(left_angles_dense)

        # Right angles
        right_angles_dense = tf.keras.layers.Dense(N_NEURONS, activation='relu', name="right_angles_dense")(last_out)
        right_angles_out = tf.keras.layers.Dense(4, activation='softmax', name="right_angles_out")(right_angles_dense)

        # Power/Precision
        power_precision_dense = tf.keras.layers.Dense(N_NEURONS, activation='relu', name="power_precision_dense")(last_out)
        power_precision_out = tf.keras.layers.Dense(2, activation='softmax', name="power_precision_out")(power_precision_dense)
        
        # Compile current model with new last layers
        self.model = tf.keras.models.Model(inputs=input_layer, 
                                           outputs=[uni_bi_out, left_right_out, left_angles_out, right_angles_out, power_precision_out], 
                                           name="multi_output")

        self.model.compile(
              loss={'uni_bi_out': 'categorical_crossentropy', 
                    'left_right_out': 'categorical_crossentropy',
                    'left_angles_out': 'categorical_crossentropy',
                    'right_angles_out': 'categorical_crossentropy',
                    'power_precision_out': 'categorical_crossentropy'
                    },
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics={
                    'uni_bi_out': 'accuracy', 
                    'left_right_out': 'accuracy',
                    'left_angles_out': 'accuracy',
                    'right_angles_out': 'accuracy',
                    'power_precision_out': 'accuracy'
                    })
        self.model.summary()
        print(self.model.output_shape)
        task()

        # Training data
        X = data_loader.X.reshape((data_loader.X.shape[0], data_loader.X.shape[1], data_loader.X.shape[2], 1))
        y = data_loader.y
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
        
        histories = {}

        split = 5
        split_i = X_train.shape[0]//split
        # Fit the model
        for _ in range(2):
            for i in range(split):
                start_i = i*split_i
                stop_i = start_i+split_i
                if stop_i > X_train.shape[0]: stop_i = -1
                history = self.model.fit(
                    X_train[start_i:stop_i],
                    [
                        y_train[start_i:stop_i,:2],
                        y_train[start_i:stop_i,2:4],
                        y_train[start_i:stop_i,4:8],
                        y_train[start_i:stop_i,8:12],
                        y_train[start_i:stop_i,12:14]
                    ],
                    steps_per_epoch=8,  
                    epochs=5,
                    verbose=1,
                    validation_data=(X_val, [
                        y_val[:,:2],
                        y_val[:,2:4],
                        y_val[:,4:8],
                        y_val[:,8:12],
                        y_val[:,12:14]
                    ]),
                    callbacks=cbs
                )
                for k in history.history:
                    if k in histories:
                        histories[k] = histories[k] + history.history[k]
                    else:
                        histories[k] = history.history[k]
        
        temp = []
        print("="*100)
        for k in histories:
            if "accuracy" in k:
                print(k)
                temp.append(histories[k])
        self.histories = np.array(temp)

        d_load, tasks = NeuroNN.CC_test()
        self.scores = []
        for task in tasks:
            # data loading from tasks
            task[0](**task[1])

            # Training data
            X = d_load.X.reshape((d_load.X.shape[0], d_load.X.shape[1], d_load.X.shape[2], 1))
            y = d_load.y
            _, X_test, _, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

            # Test the model
            predictions = self.model.predict(X_test)
            print(len(predictions))
            print(task[3])
            pred = np.argmax(predictions[task[3]], axis=1)
            print(pred.shape)
            c = 0
            for i in range(y_test.shape[0]):
                if y_test[i] == pred[i]:
                    c += 1
            print("-="*100)
            print(task[2])
            print(c/pred.shape[0])
            print("-="*100)
            self.scores.append(c/pred.shape[0]) 
            print(self.scores)

    def train_model_CC1DNN(self, task, data_loader):
        N_NEURONS = 256
        input_layer = tf.keras.layers.Input(shape=(256, 500, 1))

        # Adaptive convolutional layers
        conv1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(input_layer)
        mp1 = tf.keras.layers.MaxPooling2D(2,2)(conv1)
        conv2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(mp1)
        mp2 = tf.keras.layers.MaxPooling2D(2,2)(conv2)
        conv3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(mp2)
        mp3 = tf.keras.layers.MaxPooling2D(2,2)(conv3)
        flatten = tf.keras.layers.Flatten()(mp3)
        x1 = tf.keras.layers.Dense(1024, activation='relu', name="last_out")(flatten)
        #x2 = tf.keras.layers.Dense(512, activation='relu', name="last_out")(x1)
        o = tf.keras.layers.Dense(14, activation='relu', name="output")(x1)

        # Compile current model with new last layers
        self.model = tf.keras.models.Model(inputs=input_layer, 
                                           outputs=[o], 
                                           name="multi_output")

        self.model.compile(
                loss=['categorical_crossentropy'],
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                metrics=['accuracy'])
        self.model.summary()
        print(self.model.output_shape)
        task()

        # Training data
        X = data_loader.X.reshape((data_loader.X.shape[0], data_loader.X.shape[1], data_loader.X.shape[2], 1))
        y = data_loader.y
        X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.25, random_state=42)
        print(X_train.shape)
        print(len(y_train))
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
        histories = {}
        split = 4
        split_i = X_train.shape[0]//split
        # Fit the model
        for i in range(split):
            start_i = i*split_i
            stop_i = start_i+split_i
            if stop_i > X_train.shape[0]: stop_i = -1
            history = self.model.fit(
                X_train[start_i:stop_i],
                y_train[start_i:stop_i],
                steps_per_epoch=8,  
                epochs=25,
                verbose=2,
                validation_data=(X_val,y_val),
                callbacks=cbs
            )
            for k in history.history:
                if k in histories:
                    histories[k] = histories[k] + history.history[k]
                else:
                    histories[k] = history.history[k]
        
        temp = []
        print("="*100)
        for k in histories:
            print(k)
            temp.append(histories[k])
        self.histories = np.array(temp)

        
        d_load, tasks = NeuroNN.CC1D_test()
        self.scores = []
        for task in tasks:
            # data loading from tasks
            task[0](**task[1])

            # Training data
            X = d_load.X.reshape((d_load.X.shape[0], d_load.X.shape[1], d_load.X.shape[2], 1))
            y = d_load.y
            _, X_test, _, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

            # Test the model
            predictions = self.model.predict(X_test)
            print(predictions.shape)
            print(task[3])
            pred = np.argmax(predictions[:,task[3]], axis=1)
            print(pred.shape)
            c = 0
            for i in range(y_test.shape[0]):
                if y_test[i] == pred[i]:
                    c += 1
            print("-="*100)
            print(task[2])
            print(c/pred.shape[0])
            print("-="*100)
            self.scores.append(c/pred.shape[0]) 
            print(self.scores)

        predictions = self.model.predict(X_val)
        bi_uni_c = 0
        L_R_c = 0
        L_A_c = 0
        R_A_c = 0
        P_P_c = 0
        # bimanual/unimanual
        pred = np.argmax(predictions[:,:2], axis=1)
        y_val_t = np.argmax(y_val[:,:2], axis=1)
        for j in range(y_val_t.shape[0]):
            if y_val_t[j] == pred[j]:
                bi_uni_c += 1
        
        # right/left
        pred = np.argmax(predictions[:,2:4], axis=1)
        y_val_t = np.argmax(y_val[:,2:4], axis=1)
        for j in range(y_val_t.shape[0]):
            if y_val_t[j] == pred[j]:
                L_R_c += 1
        
        # left angles
        pred = np.argmax(predictions[:,4:8], axis=1)
        y_val_t = np.argmax(y_val[:,4:8], axis=1)
        for j in range(y_val_t.shape[0]):
            if y_val_t[j] == pred[j]:
                L_A_c += 1
        
        # right angles
        pred = np.argmax(predictions[:,8:12], axis=1)
        y_val_t = np.argmax(y_val[:,8:12], axis=1)
        for j in range(y_val_t.shape[0]):
            if y_val_t[j] == pred[j]:
                R_A_c += 1
        
        # power/precision
        pred = np.argmax(predictions[:,12:14], axis=1)
        y_val_t = np.argmax(y_val[:,12:14], axis=1)
        for j in range(y_val_t.shape[0]):
            if y_val_t[j] == pred[j]:
                P_P_c += 1

        print("-="*100)
        print(bi_uni_c)
        print(L_R_c)
        print(L_A_c)
        print(R_A_c)
        print(P_P_c)
        print(X_val.shape)
        print("-="*100)



    def test_model(self, tasks, data_loader):
        for n_model, task in enumerate(tasks):
            # data loading from tasks
            task[0](**task[1])

            # Swap last layers
            current_layers = tf.keras.models.Sequential([
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(len(data_loader.classes.keys()), activation='softmax')
            ], name=task[2])
            
            # Compile current model with new last layers
            self.model = tf.keras.models.Sequential([
                self.adp,
                current_layers
            ], name="n_model_"+str(n_model))
            self.model.build()
            current_layers.load_weights("./models/"+".h5")

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
            print(self.scores[n_model], "or", c/pred.shape[0])
            print("-="*100)

if __name__ == "__main__":
    validate_class_data_paths()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"[INFO] Using GPU: {gpus[0].name}")
    else:
        print("[INFO] No GPU found. Running on CPU.")

    neuro_nn = NeuroNN()
    simp_cl, simp_tasks = NeuroNN.multioutput_task()
    neuro_nn.train_model_CC1DNN(simp_tasks, simp_cl)

    from datetime import datetime
    np.save("scores_nn_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), neuro_nn.scores)
    np.save("histories_nn_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), neuro_nn.histories)
