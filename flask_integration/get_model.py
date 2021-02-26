import pickle
import pandas as pd
import sklearn

class LoadModel:
    #Loading the model
    def __init__(self, MODEL_PATH):
        self.loaded_model=pickle.load(open(MODEL_PATH, 'rb'))

    def predict_class(self,pregnant,insulin,bmi,age,glucose,bp,pedigree):
        #initiatlise list of list
        data = [[pregnant, insulin, bmi, age, glucose, bp, pedigree]]

        #create the pandas dataframe

        df = pd.DataFrame(data, columns= ['pregnant','insulin','bmi','age','glucose','bp','pedigree'])
        new_pred = self.loaded_model.predict(df)
        return new_pred

    # Test Model

if __name__ == '__main__':
    MODEL_PATH = "/Users/Naresh_Prajapati/Documents/2021/diabetesprediction/models/logistic_reg.sav"
    model = LoadModel(MODEL_PATH)
    predicted_class = model.predict_class(6,0,33.6,50,148,72,0.627)
    print(predicted_class)