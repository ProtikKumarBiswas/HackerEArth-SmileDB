# Problem Statement

SmileDB is a corpus of more than 100,000 happy moments crowd-sourced via Amazon's Mechanical Turk.
Each worker is given the following task: What made you happy today? Reflect on the past 24 hours, and recall three actual events that happened to you that made you happy. Write down your happy moment in a complete sentence. (Write three such moments.)
The goal of the corpus is to advance the understanding of the causes of happiness through text-based reflection.
Based on the happy moment statement you have to predict the category of happiness, i.e. the source of happiness which is typically either of the following:
	['bonding', 'achievement', 'affection', 'leisure', 'enjoy_the_moment', 'nature', 'exercise'. ]

# Data Description

The training set contains more than 60,000 samples, while your trained model will be tested on more than 40,000 samples.

<table>
	<tr>
		<th> Column Name </th>
		<th> Column Description  </th>
		<th> Column Datatype </th>
	</tr>
	<tr>
		<td>Hmid</td>
		<td>ID of the person</td>
		<td>Int64</td>
	</tr>
		
	<tr>
		<td>Reflection_period</td>
		<td>The time of happiness</td>
		<td>Object</td>
	</tr>

	<tr>
		<td>Cleaned_hm</td>  
		<td>Happiness Statement Made</td>
		<td>Object</td>
	</tr>

	<tr>
		<td>Num_sentence</td>
		<td>No. of sentences present in the person's statement.</td>
		<td>Int64</td>
	</tr>

	<tr>
		<td>Predicted_category</td>
		<td>Source of happiness</td>
		<td>Object</td>
	</tr>
</table>

---------------------------------------------------------------------------------------------
|Column Name        | Column Description                                  | Column Datatype |
---------------------------------------------------------------------------------------------
|Hmid               | Id of the person                                    | Int64           |
|Reflection_period  | The time of happiness                               | Object          |
|Cleaned_hm         | Happiness Statement Made                            | Object          |
|Num_sentence       | No. of sentences present in the person's statement. | Int64           |
|Predicted_category | Source of happiness                                 | Object          |
---------------------------------------------------------------------------------------------

# Solution and Explanation of HackerRank2.py file(Including the tools used)

The very first task was to extract the datasets from hm_train.csv and hm_test.csv(present in 'dataset folder) and then check for integrity of the dataset(eg. checking if the ids are unique etc.).
*I have also tried some pre-processing like stemming the words and chopping the text from the sentences which did perform very well.
Then the train data was randomly splited into train and test set ( train : test :: 4 : 1 ) and vectorized dataset seperately to prevent data leakage.
*I have designed two vectorizers based on AvgWord2Vec and TFIDF Weighted Word2Vec which are stored in the folders Vectorizers. I have used these vectorizers in my first 3 submissions(But did not give me better result than TF_IDF in this case).
Now using the GridSearchCV, parameters for the ComplementNB(considered as best Machine Learning algorithm for text classification task) was tuned.
After getting the best parameter(here alpha) for ComplementNB, vectorization and traing of the whole test data was done. Followed by the vectorization and prediction of the test data.

#References

https://stackoverflow.com/a/48803361/4084039
https://www.programiz.com/python-programming/examples/remove-punctuation
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
