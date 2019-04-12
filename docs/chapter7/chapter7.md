### GD
![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-1.png)
- 给到 $\theta$ (weight and bias)
- 先选择一个初始的 $\theta^0$，计算 $\theta^0$ 的loss function(没一个参数的偏微分)
- 计算完这个vector(偏微分)，然后就可以去更新的你 $\theta$ 
- millions of parameters
- BP是一个比较有效率的算法，让你计算gradient 的vector时，可以有效率的计算出来

### Chain Rule
![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-2.png)
- 连锁影响
- BP主要用到了chain rule
### BP
![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-3.png)
- cross entropy 到total loss 
- 问题是咋样计算每一笔data的partial
#### 取出一个Neural
![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-4.png)
-  



#####  forward pass

- 这个直接计算
- $\frac{\partial z}{\partial w}$，input是什么，计算结果是什么

##### Backward pass
![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-5.png)
- $\frac{\partial a}{\partial z}$得到解决

###### 计算`$\frac{\partial c}{\partial a}###### 
![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-6.png)
- 后面有几个Neural就要加几项
- $\frac{\partial z}{\partial a}$得到解决
- $\frac{\partial c}{\partial {a}'}$，$\frac{\partial c}{\partial {a}''}$假设已经知道，那么$\frac{\partial c}{\partial z}$，就是可以算出来了

![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-7.png)

##### 换个角度来看，我们得到的新neural

![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-8.png)
- `${\sigma }'(z)$`和之前的不同，这次看做是一个常数(constant)
- `$\frac{\partial c}{\partial {a}'}$`和`$\frac{\partial c}{\partial {a}''}$`做为输入的vector


##### case1:output layer
![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-9.png)
- 问题直接是output layer,直接计算


##### case2：not output layer
![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-10.png)
![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-11.png)
- 通过 $\frac{\partial c}{\partial z_a}$ 和 $\frac{\partial c}{\partial z_b}$ 计算出$\frac{\partial c}{\partial {a}'}$ 

##### 计算 $\frac{\partial c}{\partial z_a}

##### 和 $\frac{\partial c}{\partial z_b}

##### 
- 若为output layer，很快就计算出来了
- 若为not output layer，往前推(递归)

#### Backward pass 
![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-12.png)
- 先从输出层开始计算，$\frac{\partial c}{\partial z_5}$ 和 $\frac{\partial c}{\partial z_5}$ 
- 实际上就是建立一个反向的neural network

#### Summary

![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-13.png)



