len=length(AdjClose);
x=transpose([AdjClose High Low Open Volume]);
x=x(:,len:-1:1);
Close=Close(len:-1:1);
Close=transpose(Close);
x_train=x(:,1:1000);
y_train=Close(1:1000);
n=250;
for i=6:1:n
    
net=feedforwardnet([i]);
[net,tr] = trainlm(net,x_train,y_train);
y_test=(Close(1001:1030));
output_test=net(x(:,1001:1030));
e=output_test-y_test;
ploterrhist(e);
MSE(i-5)=rms(e);
mae_perf(i-5)=mae(e);

end

for i=6:1:n
    

if MSE(i-5)== min(min(MSE))
    i1=i;
   % j1=j;
end
if mae_perf(i-5)== min(min(mae_perf))
    i2=i;
    %j2=j;
end

end
plot(MSE)
%initially lets keep one layer and see the performance and plot MSE versus
%the no. of neurons
%We will find a stagnating point (lets say <10% change)
%We will do the similar 3D plot for 2 hidden layers.
%various parameters other than MSE to judge error.