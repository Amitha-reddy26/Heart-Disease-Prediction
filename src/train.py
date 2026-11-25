"""Train and evaluate nine classifiers on a public heart disease dataset."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import joblib

NUM = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
CAT = ['Sex','ChestPainType','FastingBS','RestingECG','ExerciseAngina','ST_Slope']
TARGET = 'HeartDisease'

def prepare(df):
    df = df[NUM + CAT + [TARGET]].copy()
    df[['RestingBP','Cholesterol']] = df[['RestingBP','Cholesterol']].replace(0, float('nan'))
    return df.dropna(subset=[TARGET])

def build(model):
    prep = ColumnTransformer([
        ('numeric', Pipeline([('scale', StandardScaler()), ('impute', KNNImputer(n_neighbors=5))]), NUM),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CAT)])
    return ImbPipeline([('preprocess',prep),('smote',SMOTE(random_state=42)),('model',model)])

def models():
    return {'logistic_regression':LogisticRegression(max_iter=1000), 'naive_bayes':GaussianNB(), 'random_forest':RandomForestClassifier(n_estimators=150,random_state=42), 'gradient_boosting':GradientBoostingClassifier(random_state=42), 'extra_trees':ExtraTreesClassifier(n_estimators=150,random_state=42), 'decision_tree':DecisionTreeClassifier(random_state=42), 'knn':KNeighborsClassifier(), 'svm':SVC(probability=True,random_state=42), 'mlp':MLPClassifier(max_iter=1000,random_state=42)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Path to heart.csv from the linked dataset')
    parser.add_argument('--output',default='outputs')
    parser.add_argument('--explain',action='store_true')
    args=parser.parse_args()
    df=prepare(pd.read_csv(args.data)); X=df.drop(columns=TARGET); y=df[TARGET].astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    rows=[]; fitted={}
    for name,model in models().items():
        pipe=build(model)
        score=cross_val_score(pipe,Xtr,ytr,cv=cv,scoring='roc_auc').mean()
        pipe.fit(Xtr,ytr); fitted[name]=pipe
        rows.append({'model':name,'cv_auc':score,'test_accuracy':accuracy_score(yte,pipe.predict(Xte)),'test_auc':roc_auc_score(yte,pipe.predict_proba(Xte)[:,1])})
    results=pd.DataFrame(rows).sort_values('cv_auc',ascending=False)
    results.to_csv(out/'metrics.csv',index=False); print(results.to_string(index=False))
    best=fitted[results.iloc[0]['model']]; joblib.dump(best,out/'model.joblib')
    if args.explain:
        import shap
        import matplotlib.pyplot as plt
        prep=best.named_steps['preprocess']; model=best.named_steps['model']
        background=prep.transform(Xtr.sample(min(40,len(Xtr)),random_state=42))
        sample=prep.transform(Xte.iloc[:20])
        explainer=shap.Explainer(lambda x:model.predict_proba(x)[:,1],background,feature_names=list(prep.get_feature_names_out()),algorithm='permutation')
        values=explainer(sample); shap.plots.bar(values,show=False); plt.tight_layout(); plt.savefig(out/'shap.png',dpi=150); plt.close()

if __name__=='__main__': main()
