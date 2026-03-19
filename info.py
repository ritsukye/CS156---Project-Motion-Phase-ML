import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('biosensor_dataset_with_target.csv')

# overview of the dataset
print(df.info())

# class distribution
counts = df['Event_Label'].value_counts()
print(counts)

counts.plot(kind='bar', color = 'seagreen')
plt.title('Class Distribution of Motion Phases')
plt.xlabel('Event Label')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
