## 前向过程
### 首先介绍Stochastic Differential Equations (SDE) 随机微分方程
### 对SED进行离散化 x=f(x,t)+g(x,t)η(t)
### Euler-Maruyama method->
### x(t+dt)=x(t)+f(x(t),t)dt+g(x,t)sqrt(dt)*r  <br>r是randn_normal()
<img src="img.png" alt="img" width="500" height="300" />

### 若干公式
<img src="img_1.png" alt="img_1" width="400" height="200" />

### Fokker-Planck equation
<img src="img_2.png" alt="img_2" width="400" height="200" />

## 逆向过程
### 从形式上来看,forward_process和reverse_process都是SDE
### 只要是SDE 那么就有以下几个条性质
1.有连续的形式<br>
2.有离散的形式<br>
3.从概率角度出发,有Fokker-Planck equation<br>
4.有Fokker-Planck 的解 (i.e.P(x,t))<br>
### 其中,横线是f,波浪线是g
<img src="img_3.png" alt="img_3" width="450" height="200" />

## 代码思路
<img src="img_4.png" alt="img_4" width="450" height="200" />
