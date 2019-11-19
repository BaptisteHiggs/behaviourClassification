# Imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import seaborn as sns; sns.set()

# Importing confusion Matrices
from confusionMatrices import *

# Listing the matrices together
SVMMatrices = [M_Poly_2, M_Poly_3, M_Poly_4, M_RBF_0001, M_RBF_01, M_RBF_10, M_Sigmoid_0001, M_Sigmoid_01, M_Sigmoid_10, M_Linear_0001, M_Linear_01, M_Linear_10]
NNMatrices = [M_KNN1, M_KNN10, M_KNN100, M_KNN1000]
LRMatrices = [M_LR_C0001, M_LR_C01, M_LR_C1, M_LR_C100]

# Listing the behaviours in order
behaviourOrder = ['adjust_jewelry_scarf', 'breast_feeding', 'change_clothes', 'change_diaper', 'change_pad_tampon', 'clean_glasses', 'cover_seat_with_toilet_paper', 'deal_drugs', 'defecate', 'drink_alcohol', 'eat_food', 'exercise', 'have_solace', 'hide', 'nap', 'put_in_take_out_contacts', 'read', 'smoke', 'spy', 'squat_on_toilet', 'take_medicine', 'take_phone_call', 'talk', 'urinate', 'use_drugs', 'vandalise', 'write_notes']

# Looping through each of the matrices
for matrix in LRMatrices:
    # Converting to a dataframe
    tempDF = pd.DataFrame(matrix)

    # Changing the row & column label names
    tempDF.columns = behaviourOrder
    tempDF.index = behaviourOrder

    # Plotting the heatmap
    ax = sns.heatmap(tempDF, xticklabels=1, yticklabels=1)
    plt.tight_layout()
    plt.show()

    # a b c d e f g h i j k l m n o p q r s t u v w x y z 0 1 2 3 4 5 6 7 8 9