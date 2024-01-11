import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
data=pd.read_csv("vgsales.csv")
vg=data.groupby("Genre")["Global_Sales"].count()
data.dropna()
a=st.radio("Select the type of Graph:",('Game Sold','Relation Heaatmap','Prediction model'))
if a=='Game Sold':
    cus_col=mlt.colors.Normalize(vmin=min(vg),vmax=max(vg))
    colours = [mlt.cm.PuBu(cus_col(i)) for i in vg]
    plt.figure(figsize=(7,7))
    plt.pie(vg,labels=vg.index,colors=colours)
    cent=plt.Circle((0,0),0.5,color='white')
    fig=plt.gcf()
    fig.gca().add_artist(cent)
    plt.rc('font', size=12)
    plt.title("Top 10 Categories of Games Sold", fontsize=20)
    st.pyplot(fig)
elif a=='Relation Heaatmap':
    corre=data[['Rank','Year','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']].corr()
    fig1=sns.heatmap(corre,cmap="winter_r")
    plt.title("Correlation Heatmap")
    st.pyplot(plt.show())
else:
        x = data[["Rank", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
        y = data["Global_Sales"]
        from sklearn.model_selection import train_test_split
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(xtrain, ytrain)
        predictions = model.predict(xtest)
        sns.lineplot(data=(range(0,100),predictions),markers=True)
        st.pyplot(plt.show())
print(predictions)

st.dataframe(data)
