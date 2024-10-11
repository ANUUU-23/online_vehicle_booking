import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the External CSV Dataset
df = pd.read_csv('D:/vehicle/YourCabs_training.csv')  # Replace 'path_to_file' with actual path

# Step 2: Select Relevant Columns for Segmentation
segmentation_columns = ['vehicle_model_id', 'travel_type_id', 'online_booking', 'mobile_site_booking', 'Car_Cancellation']

# Step 3: Clean the Data by Dropping Rows with Missing Values
df_clean = df[segmentation_columns].dropna()

# Step 4: Perform KMeans Clustering and Elbow Method
inertia = []  # List to store inertia values
k_values = range(1, 11)  # Testing for k from 1 to 10

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_clean)
    inertia.append(kmeans.inertia_)  # Append the inertia for each k

# Step 5: Plot Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid()
plt.show()

# Step 6: Perform Final KMeans Clustering with Chosen k
optimal_k = 4  # You can change this based on the elbow curve observation
final_kmeans = KMeans(n_clusters=optimal_k, random_state=0)
df_clean['Segment'] = final_kmeans.fit_predict(df_clean)

# Step 7: Analyze Segments
segment_analysis = df_clean.groupby('Segment').agg({
    'vehicle_model_id': lambda x: x.value_counts().idxmax(),  # Most common vehicle model
    'travel_type_id': lambda x: x.value_counts().idxmax(),    # Most common travel type
    'online_booking': 'mean',                                # Average online bookings
    'mobile_site_booking': 'mean',                           # Average mobile bookings
    'Car_Cancellation': 'mean'                               # Average car cancellation rate
}).reset_index()

print(segment_analysis)

# Step 8: Visualize the Segments (Optional)
plt.figure(figsize=(10, 6))
sns.countplot(df_clean['Segment'])
plt.title('Segment Distribution')
plt.show()

# Step 9: Target Segment Strategy (Optional)
target_segment = df_clean[df_clean['Segment'] == df_clean.groupby('Segment')['online_booking'].mean().idxmax()]

print(f"Recommended Target Segment: \n{target_segment.head()}")
