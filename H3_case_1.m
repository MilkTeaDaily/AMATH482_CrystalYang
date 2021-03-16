% Loading the videos
load('cam3_1.mat')

numFrames = size(vidFrames3_1,4);
for j = 1:numFrames
    X = vidFrames3_1(:,:,:,j); % just capture one frame of the graph
    gray = rgb2gray(X);
%     por = gray(:,320:400); % for cam1
%     por = gray(:,200:300); % for cam2
    por = gray(200:400,200:end); % for cam3
    [~,idx] = max(por(:));
    [row,col] = ind2sub(size(por),idx);
    x_3_1(j) = col+200;
    y_3_1(j) = row+200;
% Use the following code to check whether the point is on the object 
%     imshow(X); drawnow,hold on
%     plot(200+col,200+row,'r.');
%     pause(0.5);
end

%% trim the dimension to be the same
x_2_1 = x_2_1(50:end-9);
y_2_1 = y_2_1(50:end-9);
x_3_1 = x_3_1(7:end);
y_3_1 = y_3_1(7:end);
plot(1:226,y_1_1,"m-"), hold on
plot(1:226, y_2_1,"k-")
plot(1:226,y_3_1,"g-")
%% aggregate into one matrix to apply pca
d_1 = [x_1_1;y_1_1;x_2_1;y_2_1;x_3_1;y_3_1];
[U_1,S_1,V_1] = svd(d_1,'econ');
%% PLOT the SVD 
figure(1)
subplot(2,2,1)
plot(diag(S_1),'o',"MarkerSize",10);
title("Singular value of Case 1")
subplot(2,2,2)
plot(V_1(:,1:2));
title("Motion of Dominant Component Case 1")
subplot(2,2,3)
plot(diag(S_2),'o',"MarkerSize",10);
title("Singular value of Case 2")
subplot(2,2,4)
plot(V_2(:,1:2));
title("Motion of Dominant Component Case 2")
figure(2)
subplot(2,2,1)
plot(diag(S_3),'o',"MarkerSize",10);
title("Singular value of Case 3")
subplot(2,2,2)
plot(V_3(:,1:2));
title("Motion of Dominant Component Case 3")
subplot(2,2,3)
plot(diag(S_4),'o',"MarkerSize",10);
title("Singular value of Case 4")
subplot(2,2,4)
plot(V_4(:,1:2));
title("Motion of Dominant Component Case 4")
%% pca
[coeff_1,score_1,latent_1,explained_1,mu_1] = pca(pca_1);
%%
gray = rgb2gray(X);
por = gray(200:end,300:400);
[~,idx] = max(por(:));
[row,col] = ind2sub(size(por),idx);
gray_1 = imregionalmax(gray);

imshow(gray), hold on
plot(300+col,200+row,'.','Linewidth',10) % how to find the correct position? 
%plot(356,260,'r.')
%%
hsv = rgb2hsv(X(200:end,300:400,:));
v = hsv(:,:,3);
max_v = max(max(v));
[r, c] = find(v == max_v);
%% Find the brightest part on the image
S = sum(X,3);
[~,idx] = max(S(:));
[rows columns] = find(S == max(S(:)));
[row,col] = ind2sub(size(S),idx);
% What is observation and variable in this case? 
%    Observation is each time, the xy coordinate; the variable is the six
%    dimensions we have.
% variable = N = 1,2,3 cameras and we have six dimensions in total (the PCA
%   will tell us which dimensions are the most important.
% observations = ***How to extract the observation from the mat data

% Useful resources
% https://www.mathworks.com/help/stats/principal-component-analysis-pca.html