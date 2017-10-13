function out_images=scale_images(images,scale=0.5)
  pkg load image% load image package in octave
  digit_images=reshape(images,sqrt(size(images,1)),sqrt(size(images,1))*size(images,2)  );
  small_img=[];
  section=8000;
  for i=1:size(images,2)/section
    small_images=imresize(digit_images(:,(i-1)*sqrt(size(images,1))*section+1:i*sqrt(size(images,1))*section),scale);
    small_img=[small_img,small_images];
  end
  if mod(size(images,2),section) ~= 0
    small_images=imresize(digit_images(:,floor(size(images,2)/section)*sqrt(size(images,1))*section+1:end),scale);
    small_img=[small_img,small_images];
  end
  out_images=reshape(small_img,size(small_img,1)^2,size(small_img,2)/size(small_img,1));
  clear digit_images;
  clear small_images;
  clear small_img;
end