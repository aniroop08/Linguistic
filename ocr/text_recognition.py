import easyocr

def predict(image):
    reader = easyocr.Reader(['en'])
    prediction = reader.readtext(image)
    
    return prediction

def get_Text(image):
    prediction = predict(image)
    
    result = ""
    for text in prediction:
        result += text[1] + " "
       
    return result[:-1]