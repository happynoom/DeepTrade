利用LSTM网络和风险估计损失函数做股票交易
===

## 收益率
    
首先说一下收益率。离开收益率去谈算法都是在耍流氓。项目利用最近700个交易日做验证，700个交易日之前的所有交易数据作为训练集，也就是说，确保不存在验证信息的泄漏。在上证综合指数上最近700个交易日的复利收益率为200%以上，同期上证指数上涨约50%。在深圳成指近700个交易日的复利收益率也在200%左右。收益为验证收益，并非实际交易数据，且为复利计算，没有扣除交易印花税和佣金。（提醒：项目仅用于学术交流，投资有风险，损失需自负。）


## 创新点

问题定义：股票交易问题，已知一只股票的所有历史数据，包括价格、成交量、成交额等信息，在当前环境下，选择一种策略（买、卖、持、等），最大化未来的收益率。

对分类的否决。一些方法将该问题转化为分类问题，例如预测未来涨还是跌。对此提出一个极端的例子：假如有一个分类器，能够90%的准确率预测股市的涨跌（事实上70%准确都很困难），那么它10次预测涨，预测对了9次，每次涨0.1%，错的那次跌了10%，那如果我按照这个分类器来操作，预测涨的时候我都持有股票，那我将损失10%-9*0.1%=9.1%。这是不可接受的，也就是说一个90%准确的分类器都不能保证投资人赚钱。

对回归的否决：另一些方法将该问题转化为回归问题，也就是不只是预测涨跌，还要估计涨跌率。还是举一个极端的例子对这样的方法提出一些质疑：假如模型对未来的涨跌率预测值为+1%，也就是说预测为涨1%，而实际涨了8%，此时因为采信模型，我们会买入（或持有）股票，第二天我们获利8%，这种情况下在我们的模型的损失函数里却是给我们添加了|1%-8%|^2的损失，实际上模型这个错误犯得影响力很小，因为起码方向对了，可是体现在模型上我们却会惩罚模型参数；而另一种情况，模型预测值为+1%，第二天跌了6%，我们投资损失惨重，此时体现在模型的损失函数里仍然是|1%+6%|^2，两种情况对模型的惩罚程度一样，显然不太合理，我们从内心能够容忍第一种错误，而坚决不能忍第二种错误。

采用风险估计损失函数：
投资决策者并不太关心模型预测的准不准，而只关心策略赚不赚，不准也能赚。可以寻找一种损失函数的定义方法，将其直接定义为我们投资的损失率。令网络输出(p)为当天持有股票到明天的仓位，区间[0.0,1.0]，0.0表示我今天不持有任何股票至明天，1.0表示我将所有的钱买股票持有到明天，则0.5表示我用一半的钱买股票并持有到明天，那么当明天来临，股票价格的变化率(r)已经知道，那么前一个交易日的策略的收益或损失率就可以计算了：r*p，例如我今天的网络输出是p=0.4，我将40%的钱买了股票，明天这支股票涨了0.01，那么我就赚了0.01 * 0.4，如果明天跌了0.02，那我就损失了0.02 * 0.4。因此将网络的损失函数定义为：
	Loss = -100. * mean(P * R)
P为网络输出的集合，也就是持仓策略的集合，R为相应的第二天的价格变化率的集合。另外，我们知道资金是有使用成本的，如果钱不用来买股票，放在银行里也能得到利息，所以用于买股票的钱是有代价的，该成本(c)应该计算进损失函数，所以我们将损失函数重新定义为：
	Loss = -100. * mean(P * (R - c))



（事实上，由于学识有限，我也不知道是否有人采用过类似的方法，如有雷同，敬请指点。）

## 网络结构

输入层->LSTM(含DropoutWrapper)->Dense层->激活函数->输出

## 环境需要

ta-lib, ta-lib for python, numpy, tensorflow

## 版权声明

开源版本对学术应用完全免费，使用时请引用出处；商业应用需要获得授权。


## 致谢

感谢chenli0830(李辰)贡献的宝贵代码和慷慨捐赠！

Thanks to chenli0830(Chen Li) for his valuable source code and donation!



A LSTM model using Risk Estimation loss function for trades in market
===

## Introduction

   Could deep learning help us with buying and selling stocks in market? The answer could be 'Yes'. We design a solution, named DeepTrade, including history data representation, neural network construction and trading optimization methods, which could maximizing our profit based on passed experience.

   In our solution, effective representations are extracted from history data (including date/open/high/low/close/volume) first. Then a neural network based on LSTM is constructed to learn useful knowledges to direct our trading behaviors. Meanwhile, a loss function is elaborately designed to ensure the network optimizing our profit and minimizing our risk. Finaly, according the predictions of this neural network, buying and selling plans are carried out.

## Feature Representation

   History features are extracted in the order of date. Each day, with open/high/low/close/volume data, invariant features are computed, including rate of price change, MACD, RSI, rate of volume change, BOLL, distance between MA and price, distance between volume MA and volume, cross feature between price and volume. Some of these features could be used directly. Some of them should be normalized. And some should use diffrential values. A fixed length(i.e., 30 days) of feature is extracted for network learning.

## Network Construction

   LSTM network [1] is effective with learning knowleges from time series. A fixed length of history data (i.e., 30 days) is used to plan trade of next day. We make the network output a real value (p) between 0 and 1, which means how much position (percentage of capital) of the stock we should hold to tomorrow. So that if the rate of price change is r next day, out profit will be p*r. If r is negtive, we lost our money. Therefore, we define a Loss Function (called Risk Estimation) for the LSTM network:

   Loss = -100. * mean(P * R)

P is a set of our output, and R is the set of corresponding rates of price change. Further more, we add a small cost rate (c=0.0002) for money occupied by buying stock to the loss function. Then the loss function with cost rate is defined as follows:
   
   Loss = -100. * mean(P * (R - c))

  Both of these two loss functions are evaluated in our experiments.

  Our network includes four layers: LSTM layer, dense connected layer, batch normalization [3] layer, activation layer. LSTM layer is used to learn knowldges from histories. The relu6 function is used as activation to produce output value.  

## Trading Plans

   Every day, at the time before market close (nearer is better), input history features into the network, then we get an output value p. This p mean an advice of next-day's position (percentage of capital). If p=0, we should sell all we have before close. If p is positive, we should keep a poistion of p to next day, sell the redundant or buy the insufficient.

## Experimental Results

   If the network goes crazy(overfitting), just restart it. Or, a dropout layer [2] is good idea. Also, larger train dataset will help.
 
   For more demos of the experimental results, visit our website: http://www.deeplearning.xin.
   
   [Experimental Results](http://www.deeplearning.xin)
   
## Requirements

ta-lib, ta-lib for python, numpy, tensorflow

## Licence

The author is Xiaoyu Fang from China. Please quot the source whenever you use it. This project has key update already. Contact happynoom@163.com to buy a licence.

## Bug Report

Contact happynoom@163.com to report any bugs. QQ Group:370191896

## Reference

[1] Gers F A, Schmidhuber J, Cummins F, et al. Learning to Forget: Continual Prediction with LSTM[J]. Neural Computation, 2000, 12(10): 2451-2471.

[2] Srivastava N, Hinton G E, Krizhevsky A, et al. Dropout: a simple way to prevent neural networks from overfitting[J]. Journal of Machine Learning Research, 2014, 15(1): 1929-1958.

[3] Ioffe S, Szegedy C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift[C]. international conference on machine learning, 2015: 448-456.

## keras version repository
https://github.com/happynoom/DeepTrade_keras

