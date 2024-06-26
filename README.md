# Social Media Analytics and Visualisations  
**This project focuses on two aspects of web and social media analytics: statistical analysis and text mining. The Python code for this is available in the Social-Media-Analytics-and-Visualisations-Notebook.ipynb file.**  

## 1a) Statistical Analysis - Social Media Analysis  
### Dataset Identification  
Chosen Dataset: Squid Game Netflix Twitter Data from Kaggle.  
The dataset is a Twitter dump for tweets containing the hashtag #squidgame. It consists of 80,019 rows and 12 columns.  

### Data Pre-Processing  
Data Pre-processing involved:  
* Removing missing values. 
* Pre-processing of the location column as it contained variations of the same location, i.e ‘London, UK’ and ‘London, United Kingdom’.
* Pre-processing of the text column using Natural Language Processing (NLP) techniques.  

### Statistical Analysis  
To perform statistical analysis on the dataset, the following research questions were answered with the aid of visualisations.  

**1. Which ten locations had the most tweets containing the hashtag #squidgame?**  
The bar plot below shows the top ten locations by the number of tweets containing the hashtag #squidgame. The majority of the tweets containing the hashtag #squidgame are generated from the US; this indicates that Squid Game was significantly most popular in the US. The is similarly represented in the heat map.  
![Bar plot and heat map to show the top ten locations with the most tweets containing the hashtag #squidgame](https://github.com/ShriyaSami/Social-Media-Analytics-and-Visualisations/blob/4c78cf34b45ac85f1249abaee1bd128341401753/1a%20-%20Top%2010%20Locations.png)  

**2. For the top two locations with the most tweets containing the hashtag #squidgame, what are the most popular sources? How do they compare?**  
The piecharts below show the usage proportion of each source in the top two locations of the US and the UK. The first piechart shows that the most popular source in US is iPhone (55.3%), followed by web app (18.3%), Android (15.7%), SocialRabbit Plugin (7.4%) and finally TweetDeck (3.3%). In comparison, the most popular source in the UK is also iPhone but at a lower proportion of 47.5%. This is as Android (25.1%) and web app (20.4%) are more popular in the UK than they are in the US. Also, SocialRabbit Plugin is not used in the UK; instead, iPad is responsible for a small proportion of tweets.  
![Piechart to show the most popular sources within the top two locations](https://github.com/ShriyaSami/Social-Media-Analytics-and-Visualisations/blob/4c78cf34b45ac85f1249abaee1bd128341401753/1a%20-%20Popular%20Sources%20in%20Top%202%20Locations.png)  
   
**3. Are users of the hashtag #squidgame, verified or unverified users?**  
To understand the proportion of verified vs unverified users of the hashtag #squidgame, a pie chart was created. The piechart shows a larger proportion of the users in the Squid Game Twitter dataset are unverified user accounts (94.6%) which possibly represents personal accounts of individuals with an interest in Squid Game.  
![Piechart to show the proportion of verified versus unverified users](https://github.com/ShriyaSami/Social-Media-Analytics-and-Visualisations/blob/4c78cf34b45ac85f1249abaee1bd128341401753/1a%20-%20Verified%20vs%20Unverified%20Users.png)  

**4. Who are the top five users of the hashtag #squidgame?**  
The bar plot below shows the top five users, by number of tweets, of the hashtag #squidgame. The top user of the #squidgame hashtag has 400 tweets in the dataset, significantly outweighing the fifth top user who has close to only 90 tweets in the dataset.  
![Bar plot to show the top five users of the hashtag #squidgame](https://github.com/ShriyaSami/Social-Media-Analytics-and-Visualisations/blob/4c78cf34b45ac85f1249abaee1bd128341401753/1a%20-%20Top%205%20Users.png)  

**5. What is the distribution of the number of characters in the tweets containing the hashtag #squidgame?**  
To understand the distribution of the number of characters in the tweets containing the hashtag #squidgame, a Kernel Density Estimate (KDE) plot was created using the text column. At the time this dataset was collected, Twitter had a 140-character limit per tweet. Therefore, the first KDE plot (left) shows that most users utilised the maximum number of characters per tweet. The second KDE plot (right) shows how the distribution of characters altered after text pre-processing was applied. As the text pre-processing cleans the data, this shows that a large of number of tweets initially contained punctuation, URLs, symbols and/or emojis.  
![Kernel Density Estimate plot to show the distribution of the number of characters in the tweets containing the hashtag #squidgame before and after text pre-processing](https://github.com/ShriyaSami/Social-Media-Analytics-and-Visualisations/blob/4c78cf34b45ac85f1249abaee1bd128341401753/1a%20-%20Distribution%20of%20Number%20of%20Characters.png)  

### Summary  
Overall, statistical analysis techniques provided useful insights on the Squid Game Netflix Twitter dataset and allowed the research questions to be answered. The insights obtained can be used for:  
* Targeted marketing strategies - to identify geographic markets based on the most popular locations.
* Platform usage insights - to identify the most popular devices to tailor content format for optimised user experiences.
* Influencer identification - to identify and leverage the most active and engaged users for influencer marketing.
* Content analysis - to refine content based on tweeting characteristics. 

## 1b) Statistical Analysis - Graph Analysis  
This section utilises Python and Gephi to perform graph analysis.   

### Dataset Identification  
Chosen Dataset: Western States Power Grid graph dataset which consists of 4941 nodes and 6594 edges. A node is a transformer and an edge exists between two nodes when a power line connection exists between the two transformers.

### Centrality Measures
To identify the most important nodes in the network, the following centrality measures were used: degree, betweenness, closeness and Eigenvector. The measures were calculated in Python. Then, Gephi was used to create visualisations.  
**NOTE: Each visualisation builds upon the previous visualisation.**

#### Visualisation of Degree Centrality (Ranked by size)  
In the graph below, the larger the node, the higher its degree centrality(i.e. the number of direct connections to the node). This produced a very dense graph. The larger nodes are likely to be larger transformers, or power stations, and hence have more connections.  
![Visualisation of the Western States Power Grid graph dataset to identify the most important nodes by degree centrality](https://github.com/ShriyaSami/Social-Media-Analytics-and-Visualisations/blob/4c78cf34b45ac85f1249abaee1bd128341401753/1b%20-%20degree.png)  

#### Visualisation of Betweenness Centrality (Ranked by colour)  
In the graph below, nodes of a lighter colour have a lower betweenness centrality value. Nodes of a darker purple colour have a higher betweenness centrality; this is evident in the graph as the darker nodes are more towards the centre of the graph. From this, it can be interpreted that the darker nodes are larger transformers, or power stations, providing power to more than one state and hence, are more in _between_.  
![Visualisation of the Western States Power Grid graph dataset to identify the most important nodes by betweenness centrality](https://github.com/ShriyaSami/Social-Media-Analytics-and-Visualisations/blob/4c78cf34b45ac85f1249abaee1bd128341401753/1b%20-%20betweenness.png)  

#### Visualisation of Closeness Centrality (Ranked by colour)  
In the graph below, nodes of a lighter colour have a lower closeness centrality value. Nodes of a darker purple colour have a higher closeness centrality value. Nodes of a darker colour are more centralised than those of a lighter colour, which tend to be towards the outer edge of the graph. The lighter colour nodes are likely to be transformers in more isolated areas of the Western States as they have less closeness to other nodes.  
![Visualisation of the Western States Power Grid graph dataset to identify the most important nodes by closeness centrality](https://github.com/ShriyaSami/Social-Media-Analytics-and-Visualisations/blob/4c78cf34b45ac85f1249abaee1bd128341401753/1b%20-%20closeness.png)  

#### Visualisation of Eigenvector Centrality (Ranked by colour)  
In the graph below, nodes with a lower eigenvector centrality value are of a lighter colour, and vice versa for nodes with a higher eigenvector centrality. Only a handful of nodes are the darkest shade of purple and they are placed close together. This indicates that these influential nodes are also very interconnected. These nodes could be large power stations that affect power connections and hence, are more influential than other nodes.  
![Visualisation of the Western States Power Grid graph dataset to identify the most important nodes by Eigenvector centrality](https://github.com/ShriyaSami/Social-Media-Analytics-and-Visualisations/blob/4c78cf34b45ac85f1249abaee1bd128341401753/1b%20-%20eigenvector.png)  

### Community Detection
Community detection was performed to identify communities within the graph. This was done using the graph's modularity; modularity is a measure of the density of connections. As shown in the graph below, nodes have been coloured based on their modularity score. The nodes were visually separated based on their colour, creating 9 distinct communities. Based on the data, these 9 distinct communities are likely to be different sub-sections of the Western States power grid. Within each community, the nodes (i.e. transformers) are connected to each other via power lines. Some of the nodes from each community are then connected to nodes of other communities, creating a complete, connected power grid.  
![Visualisation of the Western States Power Grid graph dataset to identify communities](https://github.com/ShriyaSami/Social-Media-Analytics-and-Visualisations/blob/4c78cf34b45ac85f1249abaee1bd128341401753/1b%20-%20community%20detection.png)  

### Summary
To summarise, network analysis techniques provide useful insights into the structure of a network and the centrality of nodes. By applying these techniques to the Power Grid dataset, the most important nodes of the network were identified based on different centrality measures. Furthermore, 9 distinct communities were successfully detected and visualised.  

## 2a) Text Mining - Sentiment Analysis
To perform sentiment analysis, the pre-processed Squid Game Netflix Twitter Data from the previous section was used.  

### Lexicon-Based Approach
Initially, a lexicon-based approach was used to calculate the polarity and subjectivity of the text. The joint plot below shows the distribution of tweets based on polarity and subjectivity scores and, the colour of the data points indicates the sentiment of each tweet.  
![Distribution of tweets based on polarity and subjectivity scores](https://github.com/ShriyaSami/Social-Media-Analytics-and-Visualisations/blob/4c78cf34b45ac85f1249abaee1bd128341401753/2a%20-%20lexicon-based%20joint%20plot.png)  

It can be interpreted that tweets with polarity scores above 0 have a positive sentiment and those below 0 have a negative sentiment. A large amount of the positive tweets have a subjectivity ranging from 0.2 to 0.8, indicating a variation of facts and opinions in positive tweets. The number of positive and negative tweets that are also highly subjective is almost equal. However, when compared to the positive tweets, there are fewer negative tweets with lower subjectivity scores. This indicates that generally negative tweets include less fact and greater opinion.  

### Machine-Learning Approach - Multi-class Classification  
**Data Pre-Processing:** label encoding on target column (negative, neutral, or positive).  
**Feature Extraction:** TF-IDF (Term Frequency - Inverse Document Frequency).  
**Data Splitting:** 70:30 Train Test split.  
**Machine Learning Model:** Decision Tree classifier.  
**Model Evaluation:**  The confusion matrix below shows how the decision tree classifier predicted each class in terms of true positives, true negatives, false positives, and false negatives. The decision tree classifier performed well in predicting the true positives and true negatives across all three classes, i.e. the correct class; this indicates possible overfitting. This is further supported by the high accuracy of 98%. 
![Confusion matrix of decision tree multi-class classification](https://github.com/ShriyaSami/Social-Media-Analytics-and-Visualisations/blob/4c78cf34b45ac85f1249abaee1bd128341401753/2a%20-%20multi-class%20classification%20confusion%20matrix.png)  

### Machine-Learning Approach - Binary Classification
**Data Pre-Processing:** label encoding on target column (negative or positive).  
**Feature Extraction:** TF-IDF (Term Frequency - Inverse Document Frequency).  
**Data Splitting:** 70:30 Train Test split.  
**Machine Learning Model:** Decision Tree classifier.  
**Model Evaluation:** Again, as the decision tree classifier achieved a high accuracy of 96%, there is a possibility of overfitting. The confusion matrix below shows that the decision tree classifier did well in predicting the true positives and true negatives, i.e. the correct classes.   
![Confusion matrix of decision tree binary classification](https://github.com/ShriyaSami/Social-Media-Analytics-and-Visualisations/blob/4c78cf34b45ac85f1249abaee1bd128341401753/2a%20-%20binary%20classification%20confusion%20matrix.png)  

### Neural Network - A TensorFlow model
**Model Evaluation:** The confusion matrix shows that the model performed best in predicting the true negatives which is when a tweet with negative sentiment is predicted as negative. This could mean that tweets with negative sentiment are more easily identified than those with positive sentiment; this could be due to the use of stronger words.  
![Confusion matrix for TensorFlow neural network](https://github.com/ShriyaSami/Social-Media-Analytics-and-Visualisations/blob/4c78cf34b45ac85f1249abaee1bd128341401753/2a%20-%20neural%20network%20confusion%20matrix.png)

The loss and accuracy of the model was also plotted. For both training and testing, the model loss started greater than where it ended, showing that the error was minimised through each epoch. For both training and testing, the accuracy had a sharp incline at the first epoch.  
![Loss and accuracy plots for TensorFlow neural network](https://github.com/ShriyaSami/Social-Media-Analytics-and-Visualisations/blob/4c78cf34b45ac85f1249abaee1bd128341401753/2a%20-%20neural%20network%20evaluation.png)

### Summary 
Overall, both lexicon-based and machine learning based approaches can be for sentiment analysis. Within the lexicon-based approach, visualisations can be used to distinctly categorise tweets based on sentiment. However, the machine learning based approach is more sophisticated as the trained models could be used to predict sentiments of even larger datasets. To further improve the machine learning models, further pre-processing could be implemented as the models are prone to overfitting. 

## 2b) Text Mining - Topic Modelling
Topic modelling was implemented using Python’s unofficial News API library for the query ‘squidgame’. Descriptive analysis was performed to get a better insight on the data. The following models were used: Truncated SVD (Singular Value Decomposition), LSI (Latent Semantic Index) and LDA (Latent Dirichlet Allocation). Each of these models grouped words into ‘topics’. The word clouds below show the words as grouped into four topics by the LDA model.  
![Topic word clouds generated by LDA model](https://github.com/ShriyaSami/Social-Media-Analytics-and-Visualisations/blob/4c78cf34b45ac85f1249abaee1bd128341401753/2b%20-%20lda%20model%20word%20clouds.png)

The plots below show the perplexity and coherence of the LDA model. The perplexity has a sharp decrease but then steadily increases, indicating the model’s performance improving on unseen data. The coherence varies implying that some topics are more consistent than others.
![Perplexity and coherence of the LDA model](https://github.com/ShriyaSami/Social-Media-Analytics-and-Visualisations/blob/4c78cf34b45ac85f1249abaee1bd128341401753/2b%20-%20lda%20model%20evaluation%20.png)

### Text Summarisation 
Extraction-based summarisation was implemented to summarise a web page of information and diagrams into a more concise, single paragraph.

### Summary
To conclude, all three models returned the most prevalent topics per document. The additional coherence and perplexity scores provide a better understanding on consistency within a topic and the model’s performance on unseen data. Moreover, extraction-based summarisation was successfully implemented to summarise a lengthy web page, demonstrating how topic modelling can be effectively leveraged for text documents.
