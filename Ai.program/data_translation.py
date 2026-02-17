
import numpy as np

# Klasse navnene som er i modellen fra før
class_names = ['Glioma', 'Meningioma', 'NoTumor', 'Pituitary']

# Jeg bytter navn på klasse-navnene 
translation = {'NoTumor': "No evidence of disease (NED)",
               'Pituitary': "Low tumor burden",
               'Meningioma': "Moderate tumor burden",
               'Glioma': "High tumor burden (Advanced-stage cancer)"}

def translate_prediction(pred: np.ndarray):

    # [p_glioma, p_meningioma...] (Sannynligheten for hver klasse)
    probs = pred[0] 

    # Finner høyest sannsylighet
    idx_max = np.argmax(probs) 

    # Finner den klassen som er høyst sannynlig
    class_name = class_names[idx_max]
    main_burden = translation[class_name]

    #Procentage per translated cathegory
    burden_probs = {translation[class_names[i]]: float(probs[i] * 100.0)
                    for i in range(len(class_names))}
    
    return main_burden, burden_probs
