% Read image
I = imread('lena.jpeg');
coordinate = 100;
display("red channel");
display(I(coordinate:coordinate + 7, coordinate: coordinate+7, 1))
display("green channel");
display(I(coordinate:coordinate + 7, coordinate: coordinate+7, 2))
display("blue channel");
display(I(coordinate:coordinate + 7, coordinate: coordinate+7, 3))
figure(1);imshow(I);
hold on
rectangle('Position',[coordinate,coordinate ,8, 8],'Edgecolor', 'g', 'LineWidth', 1);
hold off
figure(2);  imshow(I(coordinate:coordinate + 7, coordinate: coordinate+7, :));  
% Convert color image to grayscale image
I_gray = rgb2gray(I);
display(I_gray(coordinate:coordinate + 7, coordinate: coordinate+7))
figure(3);  imshow(I_gray);
hold on;
rectangle('Position',[coordinate,coordinate ,8, 8],'Edgecolor', 'g', 'LineWidth', 1);
hold off;
figure(4);  imshow(I_gray(coordinate:coordinate + 7, coordinate: coordinate+7, :));
% Convert image to binary image
img_bw = im2bw(I_gray,0.5);
display(img_bw(coordinate:coordinate + 7, coordinate: coordinate+7))
figure(5);  imshow(img_bw);
hold on
rectangle('Position',[coordinate,coordinate ,8, 8],'Edgecolor', 'g', 'LineWidth', 1);
hold off
figure(6);  imshow(img_bw(coordinate:coordinate + 7, coordinate: coordinate+7, :));