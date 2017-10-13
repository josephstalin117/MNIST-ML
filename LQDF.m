%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%author: j.cai 
%%mail: jcai@mail.oom
%%date: 2016-10-11
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
% To run fast , images are resized to smaller  ones 
%images=scale_images(images); % use small images

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
img_i={};     %classified pictures
priori_i=zeros(10,1);  % a priori
u_i=zeros(size(images,1),10);  %mean
sigma_i=zeros(size(images,1),size(images,1),10); %covariance matrix for each class
sigma=zeros(size(images,1));  %overall covariance matrix
for i=0:9
  [loc,~]=find(labels==i);
  img_i{i+1}=images(:,loc);
  priori_i(i+1)=size(img_i{i+1},2)/size(images,2);
  u_i(:,i+1)=mean(img_i{i+1},2);
  x_u=img_i{i+1}-repmat(u_i(:,i+1),[1,size(img_i{i+1},2)]);
  %sigma for each class
  sigma_i(:,:,i+1)=x_u*x_u'/size(img_i{i+1},2)+0.001*eye(size(images,1));
  %calculate overall sigma
  sigma=sigma+x_u*x_u';
end
sigma=sigma/size(images,2)+0.001*eye(size(images,1));

%%test part
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

%test_images=scale_images(test_images);% use small images

test_images = V_pca' * test_images;   %the same PCA like training data 
%normalize
l2=sum(test_images.^2).^0.5+eps;
l2n = repmat(l2,size(test_images,1),1); 
test_images=test_images./l2n;
clear l2;
clear l2n;
%LDF
inv_sigma=inv(sigma);
g_i=zeros(10,size(test_images,2));
for i=0:9
  w_i=inv_sigma*u_i(:,i+1);
  w_i0=-0.5*u_i(:,i+1)'*inv_sigma*u_i(:,i+1)+log(priori_i(i+1));

  for j=1:size(test_images,2)
    g_i(i+1,j)=w_i'*test_images(:,j)+w_i0;
  end
end
[~,ldf_labels]=max(g_i);
corr_ldf=size(find(((ldf_labels-1)-test_labels')==0),2);
corr_ratio_ldf=corr_ldf/size(test_images,2);

disp(['ldf correct ratio: ',num2str(corr_ratio_ldf)]);
%QDF
inv_sigma_i={};
gq_i=zeros(10,size(test_images,2));
for i=0:9
  inv_sigma_i{i+1}=inv(sigma_i(:,:,i+1));
  wwq_i=-0.5*inv_sigma_i{i+1};
  wq_i=inv_sigma_i{i+1}*u_i(:,i+1);
  wq_i0=-0.5*u_i(:,i+1)'*inv_sigma_i{i+1}*u_i(:,i+1)+log(priori_i(i+1))-0.5*log(det(sigma_i(:,:,i+1))); %when sigma is close to be sigular,the log(det) item should be avoided
  for j=1:size(test_images,2)
    gq_i(i+1,j)=test_images(:,j)'*wwq_i*test_images(:,j)+wq_i'*test_images(:,j)+wq_i0;
  end
end
[~,qdf_labels]=max(gq_i);
corr_qdf=size(find(((qdf_labels-1)-test_labels')==0),2);
corr_ratio_qdf=corr_qdf/size(test_images,2);

disp(['qdf correct ratio: ',num2str(corr_ratio_qdf)]);