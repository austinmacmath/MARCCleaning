import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score



def filter(ceramics):

    # only ceramics with numeric values in "Form"
    ceramics = ceramics[ceramics["Form"] != "-"]
    ceramics = ceramics[ceramics["Form"] != "nan"]
    ceramics["Form"] = pd.to_numeric(ceramics["Form"], errors = "coerce")

    # drops entries in ceramics where "SITE" is nan
    ceramics["SITE"] = pd.to_numeric(ceramics["SITE"], errors = "coerce")
    ceramics = ceramics.dropna(subset = ["SITE"])
    ceramics["SITE"] = ceramics["SITE"].astype(int)

    # filters by Late Classic Ceramics
    ceramics = ceramics[ceramics["Period"] == "LC"]

    # Sorts Ceramics into Jars, Bowls, Plates, Vases
    ceramics["Vessel"] = str("Other")
    ceramics["Vessel"][(ceramics["Group"] == "JG1") | (ceramics["Group"] == "JG2") | (ceramics["Group"] == "JG3") | (ceramics["Group"] == "JG4")] = "Jars"
    ceramics["Vessel"][(ceramics["Group"] == "BG1") | (ceramics["Group"] == "BG2") | (ceramics["Group"] == "BG3") | (ceramics["Group"] == "BG4")] = "Bowls"
    ceramics["Vessel"][(ceramics["Group"] == "PG1") | (ceramics["Group"] == "PG2") | (ceramics["Group"] == "PG3") | (ceramics["Group"] == "PG4")] = "Plates"
    ceramics["Vessel"][(ceramics["Group"] == "VG1") | (ceramics["Group"] == "VG2") | (ceramics["Group"] == "VG3")] = "Vases"

    ceramics = ceramics[ceramics["Vessel"] != "Other"]

    # Sorts by the selected attributes
    ceramics2 = ceramics[["SITE", "Shape", "RimDiameter", "Pocking", "HCL", "WallThick", "Vessel"]]

    # in order to get rid of "-" values, replace "-" with "astring" and coerce them to "nan"
    ceramics2 = ceramics2.replace("-", "astring")
    ceramics2["Pocking"] = pd.to_numeric(ceramics2["Pocking"], errors = "coerce")
    ceramics2["HCL"] = pd.to_numeric(ceramics2["HCL"], errors = "coerce")
    ceramics2["WallThick"] = pd.to_numeric(ceramics2["WallThick"], errors = "coerce")

    # fills "nan" values in the selected attributes with the median of selected attributes
    # skips "nan" values in the computation of median
    ceramics2["RimDiameter"] = ceramics2["RimDiameter"].fillna(ceramics2["RimDiameter"].median(skipna = True))
    ceramics2["Pocking"] = ceramics2["Pocking"].fillna(ceramics2["Pocking"].median(skipna = True))
    ceramics2["HCL"] = ceramics2["HCL"].fillna(ceramics2["HCL"].median(skipna = True))
    ceramics2["WallThick"] = ceramics2["WallThick"].fillna(ceramics2["WallThick"].median(skipna = True))
 
    # creates a training set and a target set for the random forest
    ceramics2 = ceramics2[["SITE", "Shape", "RimDiameter", "Pocking", "HCL", "WallThick", "Vessel"]]
    return ceramics2

def build_trainer(filtered):
    train = filtered[["SITE", "Shape", "RimDiameter", "Pocking", "HCL", "WallThick"]]
    return train

def build_target(filtered):
    target = filtered["Vessel"]
    return target

def RFC(train, target):
    clf = RandomForestClassifier(n_estimators = 10, max_depth = 4)
    clf.fit(train, target)
    print ("\n")
    print ("Forest:", cross_val_score(clf, train, target, cv = 5))
    print ("\n") 

    clean = pd.read_csv("clean.csv")
    clean["RimDiameter"] = clean["RimDiameter"].fillna(train["RimDiameter"].median(skipna = True))
    clean["Pocking"] = clean["Pocking"].fillna(train["Pocking"].median(skipna = True))
    clean["HCL"] = clean["HCL"].fillna(train["HCL"].median(skipna = True))
    clean["WallThick"] = clean["WallThick"].fillna(train["WallThick"].median(skipna = True))
    clean = clean[["SITE", "Shape", "RimDiameter", "Pocking", "HCL", "WallThick"]]
    final_pred = clf.predict(clean)
    print (final_pred)
    print ("Score:", accuracy_score(pd.read_csv("clean.csv")["Vessel"], final_pred))

    return clf

def main():
    ceramics = pd.read_csv("Ceramics.csv")
    filtered_set = filter(ceramics)
    train = build_trainer(filtered_set)
    target = build_target(filtered_set)
    RFC(train, target)

    # now that we have trained a predictor, we can predict vessel type for all ceramics, including those missing information
    # next time:
    # fill ceramics with missing information with mean/median
    # predict vessel type
    # afterwards, we will have complete data set to fill in frequencies on boxplot graphs





    
    #final_pred = pd.DataFrame(data = {"Prediction": final_pred[0:]})

    #print ("Score:", accuracy_score(filtered_set["Vessel"], final_pred))


if __name__ == "__main__":
    main()