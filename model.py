import pickle
import numpy as np

def crop_pred(N,P,K,temp,humidity,ph,rainfall):


    loaded_model = pickle.load(open("model.pkl", 'rb'))
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1,-1)
    return (loaded_model.predict(single_pred))
