# ------------------------------------------Delivery4Kode--------------------------------------------------
#
# -------------------Kør python pip3 install pandas, tensorflow, numpy, scikit-learn ----------------------
# 
#  Importere relevante programsbiblioteker for at kunne køre script
import pandas as pd
import tensorflow as tf

# Loader datafilen i et Pandas Dataframe
symptom_data = pd.read_excel("Samlet_Data.xlsx")

# Importer relevant programsbibliotek til machine learning 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# Læser kolonnen ADHD i Samlet_data.xlsx 
label_encoder = preprocessing.LabelEncoder()
symptom_data['ADHD'] = label_encoder.fit_transform(symptom_data['ADHD'])

# Konverterer Pandas DataFrame til en numpy vektor
np_symptom = symptom_data.to_numpy().astype(float)

# Hiver variablerne fra øverste række (X)
X_data = np_symptom[:,1:20]

# Hiver variablen (Y) og converter den gennem one-hot-encodign, 
# hvilket vil sige at laver variablens værdi om til en boolean værdi for at kunne læse den nemmere
Y_data=np_symptom[:,20]
Y_data = tf.keras.utils.to_categorical(Y_data,2)

# Splitter data i træning sæt og test sæt
X_train,X_test,Y_train,Y_test = train_test_split( X_data, Y_data, test_size=0.10)

# Importer relevant programsbibliotek til AI
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2

# Setup til trænings parametre
EPOCHS=20
BATCH_SIZE=90
VERBOSE=1
OUTPUT_CLASSES=len(label_encoder.classes_)
N_HIDDEN=128
VALIDATION_SPLIT=0.2

# Laver en Keras sequential model
model = tf.keras.models.Sequential()

# Tilføjer Dense Layer
model.add(keras.layers.Dense(N_HIDDEN,
                             input_shape=(19,),
                              name='Dense-Layer-1',
                              activation='relu'))

# Tilføjer anden layer
model.add(keras.layers.Dense(N_HIDDEN,
                              name='Dense-Layer-2',
                              activation='relu'))

# Tilføjer en softmax layer til kategorisk prediction
model.add(keras.layers.Dense(OUTPUT_CLASSES,
                             name='Final',
                             activation='softmax'))

# Udarbjder modellen
model.compile(
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.summary()

# Bygger modellen
model.fit(X_train,
          Y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=VERBOSE,
          validation_split=VALIDATION_SPLIT)


# Evaluere modellen mod test datasættet og printer resultateterne
print("\nEvaluering af den samlede Data:\n------------------------------------")
model.evaluate(X_test,Y_test)

# Den EEG data hives ind for at predicte om personen har ADHD
print("""------------------------------------
Denne predictor kun en person ad gangen:""")
import numpy as np

Fz=416.2452163
Cz=228.5539941
Pz=351.1952054
C3=221.255983
T3=182.9553808
C4=163.5788969
T4=195.8992624
Fp1=214.960414
Fp2=269.5200531
F3=562.2617409
F4=166.9778
F7=108.6282614
F8=145.8694172
P3=162.3855928
P4=185.0662481
T5=111.3961259
T6=455.7388723
O1=164.1774267
O2=168.8988657
prediction=np.argmax(model.predict(
    [[Fz,Cz,Pz,
      C3,T3,C4,T4,Fp1,Fp2,F3,F4,F7,F8,P3,P4,T5,T6,O1,O2,]]), axis=1 )

print(label_encoder.inverse_transform(prediction))

# Predicting af flere personer på samme tid. 
# Her kan man sætte to rækker eller flere op mod hinanden.
print("""------------------------------------
Her predictes det om de to personer har ADHD.
1 = At personen har ADHD.
0 = At personen ikke har ADHD.""")
print(label_encoder.inverse_transform(np.argmax(
        model.predict([[311.7282151, 241.1301936, 160.6354176, 145.7550359,	173.5407888,	100.1834786,	156.9546314,	111.8130931,	148.9213002,	164.6445068,	271.0792034,	120.7992607,	170.4518233,	133.8991784,	212.6155055,	168.0666802,	162.9911221,	126.8945692,	132.5397214],
                                [263.8400748, 225.9962298, 334.2040816, 283.1433809, 349.4199429, 267.6733488, 472.572261,	270.2259826,	254.648011, 175.1047444,	209.9338013,	185.5657471,	312.4770791,	155.3709766,	198.9394354,	229.4747985,	282.2423319,	293.7919959,	370.6691212],
                                ]), axis=1 )))