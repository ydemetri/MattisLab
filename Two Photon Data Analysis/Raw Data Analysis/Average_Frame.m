%To get the average of image frames in a field
im=mean((M_rg),3);
im=im2double(im);
im=(im-min(im(:)))/(max(im(:))-min(im(:)));
im=imadjust(im,[0 0.2],[]);
figure;imshow(im);colormap(gray)
imwrite(im,fullfile([folder_name_wr,'average.tif']));
