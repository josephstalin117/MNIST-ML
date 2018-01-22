%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%author: josephlin
%%mail: josephlin117@gmail.oom
%%date: 2017-11-16
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

%binaryzation
th=graythresh(images);
images=imbinarize(images,th);

samp_num = size(images,2);
feature_len=size(images,1);
class_num = 10;

%%training part

prior_probability=zeros(class_num);
conditional_probability=zeros(out_num,feature_len,2);

for i=1:samp_num
    
    
    
    
    
end