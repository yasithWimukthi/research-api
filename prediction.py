import numpy
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing

# load the dataset using pandas library
dataDf = pd.read_csv("data1.csv")

leDiseases = preprocessing.LabelEncoder()
leDiseases.fit(dataDf['Diseases'])
print(leDiseases.classes_)
dataDf['Diseases'] = leDiseases.transform(dataDf['Diseases'])


leBreeds = preprocessing.LabelEncoder()
leBreeds.fit(dataDf['Breeds'])
dataDf['Breeds'] = leBreeds.transform(dataDf['Breeds'])
dataDf = dataDf.fillna(0)

model = tf.keras.models.Sequential()

def trainModel(model, datasetFilePath):
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(206, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(156, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(106, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(56, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(7, activation=tf.nn.softmax))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model_df = pd.read_csv(datasetFilePath)
    print(dataDf.shape)
    X = dataDf[['Age','Breeds','anorexia','high_fever','itching','pus_discharge','fever','dripping_urine','cough','vomiting','unable_to_swallowing','scratching','scratching_the_affected_ear_on_the_floor','blood_in_urine','eye_discharge','lethargy','patchy_hair_loss','rubbing_the_affected_ear_on_the_floor','strain_to_urinate','nasal_discharge','bloody_diarrhea','increased_salivation','crusty_skin_ulcers','frequently_rotating_movement','diarrhea','frequently_urination','mild_fever','drooling','skin_redness','rash','painful','pale_gums','licking_genital_area','abdominal_pain','seizures','inflammation_of_skin','hyperemic_gums','bloating','paralysis','unpleasant_odor','redness_on_the_ear','whining_when_urination','weight_loss','hypersensitivity','stiff_joints','muscle_twitching','weakness','odd_behavior','thickening_of_skin','abnormal_bleeding','dehydration','sores_on_the_abdomen','enlarged_lymph_nodes','sores_on_the_legs','increased_capillary_refill_time','difficulty_breathing','sores_on_the_ears','pain','sores_on_the_chest','sores_on_the_elbows','partial_paralysis','complete_paralysis']]
    y = dataDf[['Diseases']]
    x_train,x_test,y_train,y_test = train_test_split(X,y, stratify=y,test_size=0.3)

    print("==================")
    print(x_train)
    print("==================")
    print(y_train)
    print("==================")
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

    val_loss, val_acc = model.evaluate(x_test, y_test)
    print("Loss: ",val_loss)
    print("Accuracy: ",val_acc*100, "%")

    model.save('model.h5')

    print(history.history)
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return model

def predict(model, data):
    return model.predict(numpy.array(data))

trainModel(model, "data.csv")

#62
