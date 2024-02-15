from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

# Load the dataset (replace 'path_to_dataset.csv' with the actual path to your dataset)
reader = Reader(line_format='user item rating', sep=',')
data = Dataset.load_from_file('path_to_dataset.csv', reader)

# Split the data into training and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Create and train the recommendation model
algo = KNNBasic()
algo.fit(trainset)

# Get recommendations for a user (replace 'user_id' with the actual user ID)
user_id = 'user_id'
user_recommendations = algo.get_neighbors(trainset.to_inner_uid(user_id), k=10)

# Print recommended video IDs
print("Recommended video IDs for user", user_id, ":", user_recommendations)
