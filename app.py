# =============================================================================
#  ML PIPELINE DASHBOARD (COMPLETE PROJECT)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression

from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error

# ─────────────────────────────────────────────
# SESSION INIT
# ─────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 0

# ─────────────────────────────────────────────
# HEADER (HORIZONTAL STEPS)
# ─────────────────────────────────────────────
steps = [
    "Problem", "Data", "EDA", "Cleaning",
    "Features", "Split", "Model", "Train", "Metrics", "Tuning"
]

st.markdown("### 🔁 ML Pipeline Dashboard")
cols = st.columns(len(steps))

for i, s in enumerate(steps):
    if i == st.session_state.step:
        cols[i].markdown(f"**🔵 {s}**")
    else:
        cols[i].markdown(s)

# ─────────────────────────────────────────────
# STEP 0: PROBLEM TYPE
# ─────────────────────────────────────────────
if st.session_state.step == 0:

    st.header("1️⃣ Select Problem Type")

    problem = st.radio("Choose Problem Type", ["Classification", "Regression"])
    st.session_state.problem = problem

    if st.button("Next"):
        st.session_state.step += 1

# ─────────────────────────────────────────────
# STEP 1: DATA INPUT + PCA
# ─────────────────────────────────────────────
elif st.session_state.step == 1:

    st.header("2️⃣ Data Input")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df

        st.write("Shape:", df.shape)
        st.dataframe(df.head())

        target = st.selectbox("Select Target", df.columns)
        features = st.multiselect("Select Features", [c for c in df.columns if c != target])

        # PCA Visualization
        num_df = df.select_dtypes(include=np.number).dropna()

        if len(num_df.columns) >= 2:
            X_scaled = StandardScaler().fit_transform(num_df)
            pca = PCA(n_components=2)
            comp = pca.fit_transform(X_scaled)

            fig = px.scatter(x=comp[:,0], y=comp[:,1], title="PCA Projection")
            st.plotly_chart(fig)

        if target and features:
            st.session_state.target = target
            st.session_state.features = features

            if st.button("Next"):
                st.session_state.step += 1

# ─────────────────────────────────────────────
# STEP 2: EDA
# ─────────────────────────────────────────────
elif st.session_state.step == 2:

    st.header("3️⃣ EDA")

    df = st.session_state.df

    st.write(df.describe())

    col = st.selectbox("Column", df.columns)

    fig = px.histogram(df, x=col)
    st.plotly_chart(fig)

    if st.button("Next"):
        st.session_state.step += 1

# ─────────────────────────────────────────────
# STEP 3: CLEANING + OUTLIERS
# ─────────────────────────────────────────────
elif st.session_state.step == 3:

    st.header("4️⃣ Data Cleaning")

    df = st.session_state.df.copy()

    # Missing values
    method = st.selectbox("Fill Missing", ["Mean", "Median", "Mode"])

    if method == "Mean":
        df.fillna(df.mean(), inplace=True)
    elif method == "Median":
        df.fillna(df.median(), inplace=True)
    else:
        df.fillna(df.mode().iloc[0], inplace=True)

    # Outlier detection
    method_out = st.selectbox("Outlier Method", ["IQR", "IsolationForest"])

    if method_out == "IQR":
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    else:
        iso = IsolationForest()
        preds = iso.fit_predict(df.select_dtypes(include=np.number))
        df = df[preds == 1]

    st.session_state.df = df
    st.success("Cleaned")

    if st.button("Next"):
        st.session_state.step += 1

# ─────────────────────────────────────────────
# STEP 4: FEATURE SELECTION
# ─────────────────────────────────────────────
elif st.session_state.step == 4:

    st.header("5️⃣ Feature Selection")

    df = st.session_state.df
    target = st.session_state.target

    X = df.drop(columns=[target]).select_dtypes(include=np.number)
    y = df[target]

    # Variance
    vt = VarianceThreshold(0.01)
    vt.fit(X)
    kept = X.columns[vt.get_support()]

    st.write("Variance Selected:", list(kept))

    # Mutual Info
    if st.session_state.problem == "Regression":
        mi = mutual_info_regression(X, y)
        st.write("MI Scores:", dict(zip(X.columns, mi)))

    st.session_state.X = X[kept]
    st.session_state.y = y

    if st.button("Next"):
        st.session_state.step += 1

# ─────────────────────────────────────────────
# STEP 5: SPLIT
# ─────────────────────────────────────────────
elif st.session_state.step == 5:

    st.header("6️⃣ Train/Test Split")

    X = st.session_state.X
    y = st.session_state.y

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    st.success("Split Done")

    if st.button("Next"):
        st.session_state.step += 1

# ─────────────────────────────────────────────
# STEP 6: MODEL SELECTION
# ─────────────────────────────────────────────
elif st.session_state.step == 6:

    st.header("7️⃣ Model Selection")

    models = ["Linear", "SVM", "RandomForest", "KMeans"]
    model_name = st.selectbox("Model", models)

    st.session_state.model_name = model_name

    if st.button("Next"):
        st.session_state.step += 1

# ─────────────────────────────────────────────
# STEP 7: TRAIN + KFOLD
# ─────────────────────────────────────────────
elif st.session_state.step == 7:

    st.header("8️⃣ Training")

    k = st.slider("K-Folds", 3, 10, 5)

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train

    name = st.session_state.model_name

    if name == "Linear":
        model = LinearRegression()
    elif name == "SVM":
        model = SVR() if st.session_state.problem == "Regression" else SVC()
    elif name == "RandomForest":
        model = RandomForestRegressor() if st.session_state.problem=="Regression" else RandomForestClassifier()
    else:
        model = KMeans(n_clusters=3)

    scores = cross_val_score(model, X_train, y_train, cv=k)
    st.write("CV Score:", scores.mean())

    model.fit(X_train, y_train)
    st.session_state.model = model

    if st.button("Next"):
        st.session_state.step += 1

# ─────────────────────────────────────────────
# STEP 8: METRICS
# ─────────────────────────────────────────────
elif st.session_state.step == 8:

    st.header("9️⃣ Metrics")

    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    y_pred = model.predict(X_test)

    if st.session_state.problem == "Regression":
        st.metric("R2", r2_score(y_test, y_pred))
        st.metric("MSE", mean_squared_error(y_test, y_pred))
    else:
        st.metric("Accuracy", accuracy_score(y_test, y_pred))

    if st.button("Next"):
        st.session_state.step += 1

# ─────────────────────────────────────────────
# STEP 9: HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
elif st.session_state.step == 9:

    st.header("🔟 Hyperparameter Tuning")

    model = RandomForestRegressor()

    param = {
        "n_estimators": [50,100],
        "max_depth": [5,10]
    }

    grid = GridSearchCV(model, param, cv=3)
    grid.fit(st.session_state.X_train, st.session_state.y_train)

    st.write("Best Params:", grid.best_params_)
    st.write("Best Score:", grid.best_score_)