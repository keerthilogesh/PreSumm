import pandas

ip_file_1 = "C:/Users/keert/PycharmProjects/PreSumm/logs/selected_ids_False_False.pkl"
ip_file_2 = "C:/Users/keert/PycharmProjects/PreSumm/logs/selected_ids_True_False.pkl"
ip_file_3 = "C:/Users/keert/PycharmProjects/PreSumm/logs/selected_ids_False_True.pkl"

ip_df_1 = pandas.read_pickle(ip_file_1)
ip_df_2 = pandas.read_pickle(ip_file_2)
ip_df_3 = pandas.read_pickle(ip_file_3)

ip_df_1["model"] = "HISTSUMMEXT"
ip_df_2["model"] = "LEAD"
ip_df_3["model"] = "ORACLE"

ip_df = pandas.concat([ip_df_1, ip_df_2, ip_df_3], sort=True).reset_index(drop=True)

#
num_selected_sentences = 3
ip_df["selected ids"] = ip_df["selected ids"].apply(lambda x: x[:num_selected_sentences])
ip_df = ip_df.explode(column="selected ids")

#
import seaborn as sns
from matplotlib import pyplot
sns.countplot(data=ip_df, x="selected ids", hue="model")
pyplot.show()
