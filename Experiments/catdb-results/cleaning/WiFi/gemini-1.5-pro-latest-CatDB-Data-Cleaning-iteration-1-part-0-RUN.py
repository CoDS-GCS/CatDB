# ```python
import pandas as pd
original_data = 'original_data.csv'
clean_data_path = 'clean_data.csv'

df = pd.read_csv(original_data)

df['Inhibit'] = df['Inhibit'].str.lower()
df['DataPlan'] = df['DataPlan'].str.lower()
df['HotspotFrequency'] = df['HotspotFrequency'].str.lower()
df['HotspotUse'] = df['HotspotUse'].str.lower()
df['Dorm'] = df['Dorm'].str.lower()
df['Problem'] = df['Problem'].str.lower()
df['Locations'] = df['Locations'].str.lower()

df['DataPlan'] = df['DataPlan'].replace({'i don\'t have access to cellular data.': 'no'})
df['HotspotFrequency'] = df['HotspotFrequency'].replace({'i do not have a phone': 'never'})
df['HotspotUse'] = df['HotspotUse'].replace({'i do not use hotspots': 'no'})

df['Dorm'] = df['Dorm'].replace({'jane': 'jan'})

df['Locations'] = df['Locations'].replace({'working on my laptop ': 'dorm', 'the library': 'library', 'class': 'classroom', 'hodgson or student center ': 'hodgson hall', 'in the clasroom or in the library': 'classroom', 'a&t': 'a&t building', 'dorm. tv room.': 'dorm', 'in the library or at the a&t. ': 'library', 'hodgsons hall': 'hodgson hall', 'classrooms?': 'classroom', 'hight school and a&t': 'high school', 'hodgson ': 'hodgson hall', 'school, dorm': 'school', 'classrooms': 'classroom', 'a&t / dorm': 'a&t building', 'a n t': 'a&t building', 'class room': 'classroom', 'hodgson hall': 'hodgson hall', 'in class': 'classroom', 'highschool and a&t': 'high school', 'school and dorm': 'school', 'high school': 'high school', 'dorm and high school ': 'high school', 'everywhere ': 'everywhere', 'george dorm': 'george', 'library ': 'library', 'hodgenson hall': 'hodgson hall', 'upper school ': 'high school', 'hodgson hall or student center': 'hodgson hall', 'teacher learning center': 'tlc', 'hodgson/ a&t': 'hodgson hall', 'classes/library': 'library', 'the a&t': 'a&t building', 'in class': 'classroom', 'hodgson': 'hodgson hall', 'the dorm': 'dorm', 'hodson hall': 'hodgson hall', 'classes - hodgson ': 'classroom', 'classes, student center': 'classroom', 'hodgision hall': 'hodgson hall', 'in the highschool': 'high school', 'a&t building': 'a&t building', 'classroom ': 'classroom', 'hodgson hall ': 'hodgson hall', 'tlc, hodgeson hall, a and t': 'tlc', 'dorm, student center, classroom': 'dorm', 'dorn': 'dorm', 'sc/library': 'library', 'at hodgson': 'hodgson hall', 'class': 'classroom', 'at school': 'school', 'the a&t': 'a&t building', 'hogdson hall': 'hodgson hall', 'yes': 'other', 'in classrooms and in the library.': 'classroom'})
original_data = df
df.to_csv(clean_data_path, index=False)
# ```