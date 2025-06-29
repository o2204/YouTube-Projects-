

def GetPrediction(model, data: list):

    result = {
        0: "Iris-setosa",
        1: "Iris-versicolor",
        2: "Iris-virginica"}

    predections = model.predict([data])[0]
    return result[predections]
