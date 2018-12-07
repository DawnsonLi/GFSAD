# GFSAD 
## Genaral and fast supervised anomaly detection for KPI data
## 注意attention <br>
* 数据只存储了一份，在ours/data目录下，其他算法也需要这些数据，使用这些算法时，需要改变数据的位置。程序中数据文件的路径需要修改 <br>
* the data has only one copy, which is located in ours/data, and other algorithms also needs the data. the path of data in code needs to be corrected.<br>
## 介绍<br>
* 实际运维中，涉及到数量众多的KPI，例如，一个机房有1000台机器，每台机器有10条监控KPI，那么我们需要实时的监测1000*10条KPI数据，如此多的KPI，依赖人工逐一进行异常算法的选择和配置是不可行的。此外，KPI数据形状各异，每个KPI有不同的特点，需要根据KPI的特性选择合适的算法，并配置合适的参数。而且，我们需要算法不能有太高的计算复杂性，从而满足实际运维对于异常探测速度的要求，如果算法异常探测效果很好但是计算复杂度过高，不能满足探测时效性，也不能接受。
* 这里给出了一个通用的有监督学习框架，不针对某类KPI数据。通过提取KPI数据的特征，框架能够针对每个KPI的特点进行相应的学习，从而为每个KPI实现自动化异常探测，经过多个真实KPI数据的验证以及与其他几个成熟系统和算法的对比，证实我们提出的方法具有很好的通用性和实用性。<br>
* 关注实际运维问题对于时效性的要求，做了一些工作：加入了时效窗口，重新定义了精度、召回率的指标，从而评判算法的性能；给出了多个计算复杂度很低但是很有效的特征；提出了基于内存存储的数据结构帮助实现快速高效的特征提取。<br>
此外，还探讨了KPI异常探测问题中，缺失值的处理和不平衡数据的处理方法，我们进行了多个实验，以发现每种处理方法的特点和规律。<br>
### KPI数据类型
我们的数据涉及三大类KPI，包括平稳型，不平稳型，周期型。我们的方法具有通用性，不需要提前判断KPI的类别。 <br>
* 平稳型
![平稳型KPI](https://github.com/DawnsonLi/GFSAD/blob/master/pic/stable.png)
* 不平稳型
![不平稳型KPI](https://github.com/DawnsonLi/GFSAD/blob/master/pic/unstable.png)
* 周期型
![周期型KPI](https://github.com/DawnsonLi/GFSAD/blob/master/pic/seasonal.png)
