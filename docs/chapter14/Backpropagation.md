### GD
![image](80B89D7CF6914912A7D6C7E7D2304E54)
- 给到`$\theta$`(weight and bias)
- 先选择一个初始的`$\theta^0$`，计算`$\theta^0$`的loss function(没一个参数的偏微分)
- 计算完这个vector(偏微分)，然后就可以去更新的你`$\theta$`
- millions of parameters
- BP是一个比较有效率的算法，让你计算gradient 的vector时，可以有效率的计算出来

### Chain Rule
![image](4ECB6CCE4A874E528B113BB718EB912C)
- 连锁影响
- BP主要用到了chain rule
### BP
![image](6FC03F07201F4792BD00049068E7BA81)
- cross entropy 到total loss 
- 问题是咋样计算每一笔data的partial
#### 取出一个Neural
![image](FD3C2789C60847A583463671315D11AA)
-  



#####  forward pass

- 这个直接计算
- `$\frac{\partial z}{\partial w}$`，input是什么，计算结果是什么
 
##### Backward pass
![image](38AF86AB49C14D87BC41821F3319423C)
- `$\frac{\partial a}{\partial z}$`得到解决

###### 计算`$\frac{\partial c}{\partial a}$`
![image](A0B302DDC96948308574DA38E8F866C8)
- 后面有几个Neural就要加几项
- `$\frac{\partial z}{\partial a}$`得到解决
- `$\frac{\partial c}{\partial {a}'}$`，`$\frac{\partial c}{\partial {a}''}$`假设已经知道，那么`$\frac{\partial c}{\partial z}$`，就是可以算出来了

![image](B8C719C2379748F19541C9D6C5ABEF63)

##### 换个角度来看，我们得到的新neural

![image](2F9830B6FB5B4685923ECA639C826A7C)
- `${\sigma }'(z)$`和之前的不同，这次看做是一个常数(constant)
- `$\frac{\partial c}{\partial {a}'}$`和`$\frac{\partial c}{\partial {a}''}$`做为输入的vector


##### case1:output layer
![image](02FD6523A0D940CE82203C0C7F5DC140)
- 问题直接是output layer,直接计算


##### case2：not output layer
![image](B3AE88B7739A4718BDB99E600DD34F7F)
![image](050343294B0B45DC905CF88C15DC0843)
- 通过`$\frac{\partial c}{\partial z_a}$`和`$\frac{\partial c}{\partial z_b}$`计算出`$\frac{\partial c}{\partial {a}'}$`

##### 计算`$\frac{\partial c}{\partial z_a}$`和`$\frac{\partial c}{\partial z_b}$`
- 若为output layer，很快就计算出来了
- 若为not output layer，往前推(递归)

#### Backward pass 
![image](A37A40F27E4C43AFB5D872235DD841F2)
- 先从输出层开始计算，`$\frac{\partial c}{\partial z_5}$`和`$\frac{\partial c}{\partial z_5}$`
- 实际上就是建立一个反向的neural network
 
#### Summary

![image](6F2FCCD63D5E44A09FF41EB42FE72700)



