# Pickle package
import pickle

# load the model from disk
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

def irrigate_or_not_irrigate(soil_moisture):
    if(loaded_model.predict([[soil_moisture]]))==0:
        print('Irrigate')
    else:
        print('Not irrigate')

