%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%author: josephlin
%%mail: josephlin117@gmail.oom
%%date: 2017-11-16
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

% PCA dimensionality reduction
C = double(images * images');
[V, D] = eig(C);
D = diag(D); % perform PCA on features matrix 
D = cumsum(D) / sum(D);
k = find(D >= 1e-2, 1); % ignore 1% energy
V_pca = V(:, k:end); % choose the largest eigenvectors' projection
clear C;
clear D;
clear V;
images = V_pca' * images;

%normalize
l2=sum(images.^2).^0.5+eps;
l2n = repmat(l2,size(images,1),1);
images=images./l2n;
clear l2;
clear l2n;

%%training part

%%get samp num
samp_num = size(images,2);
inp_num=size(images,1);
out_num = 10;
hid_num = 9;
%init ??
w1 = rand(inp_num, hid_num) * 0.2 - 0.1;
w2 = rand(hid_num,out_num) * 0.2 - 0.1;
hid_offset = zeros(hid_num,1);
out_offset=zeros(out_num,1);
inp_lrate = 0.3;
hid_lrate = 0.3;
err_th = 0.01;

for i=1:samp_num
    t_label=zeros(out_num,1);
    %9???8
    t_label(labels(i,:)+1,1)=1;
    hid_value = w1'*images(:,i)+ hid_offset;
    hid_act=zeros(size(hid_value,1),1);
    %?????
    for k=1:size(hid_value,1)
        hid_act(k,1)=1/(1+exp(-hid_value(k,1)));
    end
    out_value = w2'*hid_act + out_offset;
    out_act=zeros(size(out_value,1),1);
    for k=1:size(out_value,1)
        out_act(k,1)=1/(1+exp(-out_value(k,1)));
    end
    
    e = t_label - out_act;
    out_delta = e .* out_act .* (1-out_act);
    hid_delta = hid_act .* (1-hid_act) .* (w2 * out_delta);
    for k=1:out_num
        w2(:,k) =w2(:,k) + hid_lrate*out_delta(k,:).*hid_act;
    end
    for k=1:hid_num
        w1(:,k) =w1(:,k) + inp_lrate*hid_delta(k,:).*images(:,i);
    end
    
    %????
    out_offset = out_offset + hid_lrate * out_delta;                             
    hid_offset = hid_offset + inp_lrate * hid_delta;
    
end

%%test part
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
%PCA
test_images = V_pca' * test_images;
%normalize
l2=sum(test_images.^2).^0.5+eps;
l2n = repmat(l2,size(test_images,1),1); 
test_images=test_images./l2n;
clear l2;
clear l2n;

test_num=size(test_images,2);
right=0;

for i=1:test_num
    hid_value = w1'*test_images(:,i) + hid_offset;
        hid_act=zeros(size(hid_value,1),1);
    %?????
    for k=1:size(hid_value,1)
        hid_act(k,1)=1/(1+exp(-hid_value(k,1)));
    end
    out_value = w2'*hid_act + out_offset;
    out_act=zeros(size(out_value,1),1);
    for k=1:size(out_value,1)
        out_act(k,1)=1/(1+exp(-out_value(k,1)));
    end
    
    [~,bp_labels]=max(out_act);
    
    
    if bp_labels==test_labels(i)+1
        
        right=right+1;
    end
end

corr=right/test_num;
disp(corr);
