import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# 1) Take user input for prediction at the start
print("\n🔷 Enter diamond attributes for price prediction:")

carat = float(input("Carat (e.g., 0.75): "))
cut = input("Cut (Fair, Good, Very Good, Premium, Ideal): ").title()
color = input("Color (D to J): ").upper()
clarity = input("Clarity (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF): ").upper()
depth = float(input("Depth percentage (e.g., 61.5): "))
table = float(input("Table percentage (e.g., 55): "))
x = float(input("Length x in mm (e.g., 5.8): "))
y_dim = float(input("Width y in mm (e.g., 5.8): "))
z = float(input("Depth z in mm (e.g., 3.6): "))
size = x * y_dim * z

user_data = pd.DataFrame([{
    'carat': carat,
    'cut': cut,
    'color': color,
    'clarity': clarity,
    'depth': depth,
    'table': table,
    'size': size
}])

# 2) Load diamonds dataset
file_path = r"C:\Users\VINOD\Desktop\Diamond_Price_Analysis\diamonds.csv"
df = pd.read_csv(file_path)

# 3) Add size column if x, y, z exist
if {'x', 'y', 'z'}.issubset(df.columns):
    df['size'] = df['x'] * df['y'] * df['z']

# 4) Drop rows with missing or zero values
df = df.dropna()
df = df[(df[['carat', 'depth', 'table', 'x', 'y', 'z']] > 0).all(axis=1)]

# 5) Define features and target
features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'size']
target = 'price'

X = df[features]
y = df[target]

# 6) Visualizations AFTER user input
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='carat', y='price', hue='cut', alpha=0.6)
plt.title('Diamond Price vs Carat by Cut')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.legend(title='Cut')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='clarity', y='price', order=sorted(df['clarity'].unique()))
plt.title('Diamond Price by Clarity')
plt.xlabel('Clarity')
plt.ylabel('Price')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='color', y='price', order=sorted(df['color'].unique()))
plt.title('Diamond Price by Color')
plt.xlabel('Color')
plt.ylabel('Price')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
numerical_cols = df.select_dtypes(include='number')
sns.heatmap(numerical_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# 7) Preprocessing
categorical_features = ['cut', 'color', 'clarity']
numerical_features = ['carat', 'depth', 'table', 'size']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# 8) Train-test split & train models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100)
}

best_model_name = None
best_r2 = -float('inf')
best_model_pipeline = None

print("\n🔎 Model Performance:")

for name, reg in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', reg)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{name}: R^2 = {r2:.3f}, MAE = ${mae:,.0f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_model_pipeline = pipeline

print(f"\n✅ Best model: {best_model_name} (R^2 = {best_r2:.3f})")

# 9) Feature importance (if tree model)
if hasattr(best_model_pipeline.named_steps['regressor'], 'feature_importances_'):
    encoder = best_model_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    ohe_features = encoder.get_feature_names_out(categorical_features)
    all_features = list(ohe_features) + numerical_features
    importances = best_model_pipeline.named_steps['regressor'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x='importance', y='feature')
    plt.title(f'Feature Importances ({best_model_name})')
    plt.tight_layout()
    plt.show()

# 10) Predict price with best model
predicted_price = best_model_pipeline.predict(user_data)[0]
print(f"\n💎 Estimated Diamond Price with {best_model_name}: ${predicted_price:,.2f}")
