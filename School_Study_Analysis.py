#jupyter notebook

import numpy as np #linear algebra
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import seaborn as sns
sns.set(style='ticks')



data=pd.read_csv('studentData.csv')
print('done')


print(data.info())

data[['Class','raisedhands','VisITedResources','AnnouncementsView']].head()


melt = pd.melt(data,id_vars='Class',value_vars=['raisedhands','VisITedResources','AnnouncementsView'])
melt.head()

melt.describe()

sns.swarmplot(x='variable',y='value',hue='Class' , data=melt,palette={'H':'lime','M':'grey','L':'red'})
plt.ylabel('Values from zero to 100')
plt.title('High, middle and low level students')
plt.show()

testVar=melt[:10]
testVar


ave_raisedhands = sum(data['raisedhands'])/len(data['raisedhands'])
print(ave_raisedhands)
ave_VisITedResources = sum(data['VisITedResources'])/len(data['VisITedResources'])
print(ave_VisITedResources)
ave_AnnouncementsView = sum(data['AnnouncementsView'])/len(data['AnnouncementsView'])
print(ave_AnnouncementsView)

unsuccess = data[(data['raisedhands'] >= ave_raisedhands) & (data['VisITedResources']>=ave_VisITedResources) & (data['AnnouncementsView']>=ave_AnnouncementsView)  & (data['Class'] == 'L')]
betterThenAverageStudent = data[(data['raisedhands'] >= ave_raisedhands) & (data['VisITedResources']>=ave_VisITedResources) & (data['AnnouncementsView']>=ave_AnnouncementsView)  ]



unsuccess.head()

betterThenAverageStudent.head()

data['numeric_class'] = [1 if data.loc[i,'Class'] == 'L' else 2 if data.loc[i,'Class'] == 'M' else 3 for i in range(len(data))]
data['numeric_class'].head(16)


grade_male_ave = sum(data[data.gender == 'M'].numeric_class)/float(len(data[data['gender'] == 'M']))
grade_female_ave = sum(data[data.gender == 'F'].numeric_class)/float(len(data[data['gender'] == 'F']))



nation = data.NationalITy.unique()
nation_grades_ave = [sum(data[data.NationalITy == i].numeric_class)/float(len(data[data.NationalITy == i])) for i in nation]
ax = sns.barplot(x=nation, y=nation_grades_ave)
jordan_ave = sum(data[data.NationalITy == 'Jordan'].numeric_class)/float(len(data[data.NationalITy == 'Jordan']))
print('Jordan average: '+str(jordan_ave))
plt.xticks(rotation=90)



lessons = data.Topic.unique()
lessons_grade_ave=[sum(data[data.Topic == i].numeric_class)/float(len(data[data.Topic == i])) for i in lessons]
ax = sns.barplot(x=lessons, y=lessons_grade_ave)
plt.title('Students Success on different topics')
chemistry_ave = sum(data[data.Topic == 'Chemistry'].numeric_class)/float(len(data[data.Topic == 'Chemistry']))
print('Chemistry average: '+ str(chemistry_ave))
plt.xticks(rotation=90)


relation = data.Relation.unique()
relation_grade_ave = [sum(data[data.Relation == i].numeric_class)/float(len(data[data.Relation == i])) for i in relation]
ax = sns.barplot(x=relation, y=relation_grade_ave)
plt.title('Relation with father or mother affects success of students')



discussion = data.Discussion
discussion_ave = sum(discussion)/len(discussion)
ax = sns.violinplot(y=discussion,split=True,inner='quart')
ax = sns.swarmplot(y=discussion,color='black')
ax = sns.swarmplot(y = unsuccess.Discussion, color='red')
plt.title('Discussion group participation')


absence_day = data.StudentAbsenceDays.unique()
absense_day_ave = [sum(data[data.StudentAbsenceDays == i].numeric_class)/float(len(data[data.StudentAbsenceDays == i])) for i in absence_day]
ax = sns.barplot(x=absence_day, y=absense_day_ave)
plt.title('Absence effect on success')
