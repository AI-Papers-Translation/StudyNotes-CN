# [Python 数据挖掘 工具包整理](https://www.cnblogs.com/to-creat/p/6559197.html)

## 连接器与io

### 数据库

| 类别    | Python                       | R                |
| ------- | ---------------------------- | ---------------- |
| MySQL   | mysql-connector-python(官方) | RMySQL           |
| Oracle  | cx_Oracle                    | ROracle          |
| MongoDB | pymongo                      | RMongo, rmongodb |
| ODBC    | pyodbc                       | RODBC            |

### IO类

| 类别  | Python                                       | R                                          |
| ----- | -------------------------------------------- | ------------------------------------------ |
| excel | xlsxWriter, pandas.(from/to)_excel, openpyxl | openxlsx::read.xlsx(2), xlsx::read.xlsx(2) |
| csv   | csv.writer                                   | read.csv(2), read.table                    |
| json  | json                                         | jsonlite                                   |
| 图片  | PIL                                          | jpeg, png, tiff, bmp                       |

## 统计类

### 描述性统计

| 类别              | Python                                                       | R               |
| ----------------- | ------------------------------------------------------------ | --------------- |
| 描述性统计汇总    | scipy.stats.descirbe                                         | summary         |
| 均值              | scipy.stats.gmean(几何平均数), scipy.stats.hmean(调和平均数), numpy.mean, numpy.nanmean, pandas.Series.mean | mean            |
| 中位数            | numpy.median, numpy.nanmediam, pandas.Series.median          | median          |
| 众数              | scipy.stats.mode, pandas.Series.mode                         | 未知            |
| 分位数            | numpy.percentile, numpy.nanpercentile, pandas.Series.quantile | quantile        |
| 标准差            | scipy.stats.std, scipy.stats.nanstd, numpy.std, pandas.Series.std | sd              |
| 方差              | numpy.var, pandas.Series.var                                 | var             |
| 变异系数          | scipy.stats.variation                                        | 未知            |
| 协方差            | numpy.cov, pandas.Series.cov                                 | cov             |
| (Pearson)相关系数 | scipy.stats.pearsonr, numpy.corrcoef, pandas.Series.corr     | cor             |
| 峰度              | scipy.stats.kurtosis, pandas.Series.kurt                     | e1071::kurtosis |
| 偏度              | scipy.stats.skew, pandas.Series.skew                         | e1071::skewness |
| 直方图            | numpy.histogram, numpy.histogram2d, numpy.histogramdd        | 未知            |

### 回归

| 类别                    | Python                                                 | R                    |
| ----------------------- | ------------------------------------------------------ | -------------------- |
| 普通最小二乘法回归(ols) | statsmodels.ols, sklearn.linear_model.LinearRegression | lm,                  |
| 广义线性回归(gls)       | statsmodels.gls                                        | nlme::gls, MASS::gls |

### 假设检验

| 类别                | Python                                                       | R        |
| ------------------- | ------------------------------------------------------------ | -------- |
| t检验               | statsmodels.stats.ttest_ind, statsmodels.stats.ttost_ind, statsmodels.stats.ttost.paired; scipy.stats.ttest_1samp, scipy.stats.ttest_ind, scipy.stats.ttest_ind_from_stats, scipy.stats.ttest_rel | t.test   |
| Pearson相关系数检验 | scipy.stats.pearsonr                                         | cor.test |

### 时间序列

| 类别  | Python                        | R     |
| ----- | ----------------------------- | ----- |
| AR    | statsmodels.ar_model.AR       | ar    |
| ARIMA | statsmodels.arima_model.arima | arima |
| VAR   | statsmodels.var_model.var     | 未知  |

### SVM(支持向量机)

| 类别                           | Python                | R          |
| ------------------------------ | --------------------- | ---------- |
| 支持向量分类器（SVC）          | sklearn.svm.SVC       | e1071::svm |
| 非支持向量分类器（nonSVC）     | sklearn.svm.NuSVC     | 未知       |
| 线性支持向量分类器(Lenear SVC) | sklearn.svm.LinearSVC | 未知       |

#### 基于临近

| 类别                                        | Python                                      | R    |
| ------------------------------------------- | ------------------------------------------- | ---- |
| k-临近分类器                                | sklearn.neighbors.KNeighborsClassifier      | 未知 |
| 半径临近分类器                              | sklearn.neighbors.RadiusNeighborsClassifier | 未知 |
| 临近重心分类器(Nearest Centroid Classifier) | sklearn.neighbors.NearestCentroid           | 未知 |

### 贝叶斯

| 类别                                | Python                            | R                 |
| ----------------------------------- | --------------------------------- | ----------------- |
| 朴素贝叶斯                          | sklearn.naive_bayes.GaussianNB    | e1071::naiveBayes |
| 多维贝叶斯(Multinomial Naive Bayes) | sklearn.naive_bayes.MultinomialNB | 未知              |
| 伯努利贝叶斯(Bernoulli Naive Bayes) | sklearn.naive_bayes.BernoulliNB   | 未知              |

### 决策树

| 类别           | Python                                  | R                                          |
| -------------- | --------------------------------------- | ------------------------------------------ |
| 决策树分类器   | sklearn.tree.DecisionTreeClassifier     | tree::tree, party::ctree                   |
| 决策树回归器   | sklearn.tree.DecisionTreeRegressor      | tree::tree, party::tree                    |
| 随机森林分类器 | sklearn.ensemble.RandomForestClassifier | randomForest::randomForest, party::cforest |
| 随机森林回归器 | sklearn.ensemble.RandomForestRegressor  | randomForest::randomForest, party::cforest |

### 聚类

| 类别     | Python                           | R               |
| -------- | -------------------------------- | --------------- |
| kmeans   | scipy.cluster.kmeans.kmeans      | kmeans::kmeans  |
| 分层聚类 | scipy.cluster.hierarchy.fcluster | (stats::)hclust |

### 关联规则

| 类别          | Python                                                       | R               |
| ------------- | ------------------------------------------------------------ | --------------- |
| apriori算法   | apriori(可靠性未知，不支持py3), PyFIM(可靠性未知，不可用pip安装) | arules::apriori |
| FP-Growth算法 | fp-growth(可靠性未知，不支持py3), PyFIM(可靠性未知，不可用pip安装) | 未知            |

### 神经网络

| 类别     | Python                | R                                |
| -------- | --------------------- | -------------------------------- |
| 神经网络 | neurolab.net, keras.* | nnet::nnet, nueralnet::nueralnet |
| 深度学习 | keras.*               | 不可靠包居多以及未知             |

#  

## 文本基本操作

 

 

| 类别      | Python                                      | R                                         |
| --------- | ------------------------------------------- | ----------------------------------------- |
| tokenize  | nltk.tokenize(英), jieba.tokenize(中)       | tau::tokenize                             |
| stem      | nltk.stem                                   | RTextTools::wordStem, SnowballC::wordStem |
| stopwords | stop_words.get_stop_words                   | tm::stopwords, qdap::stopwords            |
| 中文分词  | jieba.cut, smallseg, Yaha, finalseg, genius | jiebaR                                    |
| TFIDF     | gensim.models.TfidfModel                    | 未知                                      |