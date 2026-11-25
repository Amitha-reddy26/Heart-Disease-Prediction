import numpy as np
import pandas as pd
from src.train import prepare, build, models, NUM, CAT, TARGET

def test_pipeline():
    rng=np.random.default_rng(42)
    df=pd.DataFrame({c:rng.uniform(1,100,60) for c in NUM})
    for c in CAT: df[c]=rng.choice(['a','b'],60)
    df[TARGET]=[0]*30+[1]*30
    df.loc[0,'Cholesterol']=0
    clean=prepare(df)
    assert pd.isna(clean.loc[0,'Cholesterol'])
    pipe=build(models()['naive_bayes']).fit(clean.drop(columns=TARGET),clean[TARGET])
    assert pipe.predict_proba(clean.drop(columns=TARGET)).shape==(60,2)
    assert len(models())==9
