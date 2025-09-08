import pandas as pd
import numpy as np
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


@st.cache_data
def load_data(path='Cancer Data.csv'):
	df = pd.read_csv(path)
	# drop columns that are entirely empty
	df = df.dropna(axis=1, how='all')
	# Ensure common columns are present
	if 'id' in df.columns:
		df = df.drop(columns=['id'])
	return df


def prepare_features(df):
	# keep numeric feature columns only and the diagnosis target
	feature_cols = df.drop(columns=['diagnosis']).select_dtypes(include=["int64", "float64"]).columns.tolist()
	X = df[feature_cols]
	y = df['diagnosis'].map({'B': 0, 'M': 1})
	return X, y, feature_cols


def train_model(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)
	pipeline = Pipeline([
		('imputer', SimpleImputer(strategy='mean')),
		('scaler', StandardScaler()),
		('clf', LogisticRegression(max_iter=1000))
	])
	pipeline.fit(X_train, y_train)
	y_pred = pipeline.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	return pipeline, acc


def predict_single(pipeline, feature_names, values_dict):
	# Build a DataFrame with a single row using feature_names order
	row = pd.DataFrame([ [ values_dict.get(f, np.nan) for f in feature_names ] ], columns=feature_names)
	pred = pipeline.predict(row)[0]
	prob = pipeline.predict_proba(row)[0, 1]
	return int(pred), float(prob)


def main():
	st.set_page_config(page_title='Breast Cancer Predictor', layout='wide')
	st.title('Breast Cancer Diagnosis Demo')

	st.markdown('This app trains a simple Logistic Regression model on the provided dataset and lets you predict whether a sample is benign or malignant.')

	df = load_data()
	st.sidebar.header('Options')
	show_data = st.sidebar.checkbox('Show raw data', value=False)
	if show_data:
		st.dataframe(df.head())

	# Prepare features and train model
	X, y, feature_names = prepare_features(df)
	model_pipeline, test_acc = train_model(X, y)
	#st.metric('Test accuracy', f'{test_acc:.4f}')

	st.header('Make a prediction')
	input_mode = st.radio('Input mode', ['Manual single sample', 'Upload CSV of samples'], index=0)

	if input_mode == 'Manual single sample':
		st.write('Enter feature values (defaults = column mean)')
		# Use two-column layout for inputs
		cols = st.columns(2)
		values = {}
		for i, feat in enumerate(feature_names):
			col = cols[i % 2]
			series = X[feat]
			default = float(series.mean())
			try:
				min_val = float(series.min())
				max_val = float(series.max())
			except Exception:
				min_val = default - 1.0
				max_val = default + 1.0
			# Handle NaN or constant columns by expanding a small range
			if np.isnan(min_val) or np.isnan(max_val):
				min_val = default - 1.0
				max_val = default + 1.0
			if min_val == max_val:
				delta = abs(min_val) * 0.1 if min_val != 0 else 0.1
				min_val = min_val - delta
				max_val = max_val + delta
			v = col.slider(feat, min_value=min_val, max_value=max_val, value=default)
			values[feat] = float(v)

		if st.button('Predict'):
			pred, prob = predict_single(model_pipeline, feature_names, values)
			label = 'Malignant (1)' if pred == 1 else 'Benign (0)'
			st.write('Prediction:', label)
			st.write(f'Predicted probability of malignancy: {prob:.3f}')

	else:
		uploaded = st.file_uploader('Upload CSV file with feature columns', type=['csv'])
		if uploaded is not None:
			uploaded_df = pd.read_csv(uploaded)
			# Keep only the feature columns in correct order; if missing columns, fill with NaN
			missing = [c for c in feature_names if c not in uploaded_df.columns]
			if missing:
				st.warning(f'Uploaded file is missing columns: {missing}. Missing columns will be filled with NaN.')
				for m in missing:
					uploaded_df[m] = np.nan
			prepared = uploaded_df[feature_names]
			preds = model_pipeline.predict(prepared)
			probs = model_pipeline.predict_proba(prepared)[:, 1]
			results = prepared.copy()
			results['pred'] = preds
			results['prob_malignant'] = probs
			st.dataframe(results)

	st.sidebar.markdown('---')
	st.sidebar.write('Model: simple pipeline (mean imputation → standard scaling → logistic regression)')


if __name__ == '__main__':
	main()