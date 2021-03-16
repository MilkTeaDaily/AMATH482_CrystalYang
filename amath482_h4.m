clear all; close all; clc
% Extract the needed data
gunzip('train-labels-idx1-ubyte.gz');
gunzip('train-images-idx3-ubyte.gz');
gunzip('t10k-images-idx3-ubyte.gz');
gunzip('t10k-labels-idx1-ubyte.gz');
[train_images, train_labels] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
[test_images, test_labels] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');

%% Reshape
train_images = double(reshape(train_images, size(train_images,1)*size(train_images,2), []).');
train_labels = double(train_labels);
%% SVD 
% feature/mode = row
% image on column, 1 column is the image of digit
trainingdata = transpose(train_images);
traininglb = transpose(train_labels);
[U,S,V] = svd(trainingdata,'econ');

%% rank of digit space equals to the non zero singular value
subplot(1,2,1)
plot(diag(S),'ko')
title("Singular value")
r = rank(S);
sig = diag(S);
subplot(1,2,2)
plot(sig.^2/sum(sig.^2),'ko')
title("Engergy of the singular value")
ylabel('Energy')
%%
energy = sig.^2/sum(sig.^2);
sum(energy(1:200))
%% Selected V-mode 2,3,5
% random sample of the data -to be less to show
ind = randsample(60000,2000);
l = traininglb(ind); % label
cmap = colormap(parula(10));
proj(:,1) = U(:,2);
proj(:,2) = U(:,3);
proj(:,3) = U(:,5);
m = proj'*trainingdata(:,ind);
for j = 1:10
    plot3(m(1,find(l==(j-1))),m(2,find(l==(j-1))),m(3,find(l==(j-1))),'.','Color',cmap(j,:)), hold on
end

%% LDA of 2 digits(group)
% random sample 2000 out of the 60000
ind = randsample(60000,2000);
l = traininglb(ind); % label
feature = 200; % features we used
digits = S*V';
ind_2 = find(l == 2);
ind_3 = find(l == 3);
digit_2 = digits(1:feature,ind_2);
digit_3 = digits(1:feature,ind_3);
m2 = mean(digit_2,2);
m3 = mean(digit_3,2);
% within-class variance
Sw = 0; 
for k = 1:length(ind_2)
    Sw = Sw + (digit_2(:,k) - m2)*(digit_2(:,k) - m2)';
end
for k = 1:length(ind_3)
    Sw = Sw + (digit_3(:,k) - m3)*(digit_3(:,k) - m3)';
end
% between class
Sb = (m2-m3)*(m2-m3)'; 

[V2, D] = eig(Sb,Sw);
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);

v2 = w'*digit_2;
v3 = w'*digit_3;

if mean(v2)>mean(v3)
    w = -w;
    v2 = -v2;
    v3 = -v3;
end

plot(v2,zeros(length(ind_2)),'ob','Linewidth',2), hold on
plot(v3,ones(length(ind_3)),'dr','Linewidth',2)

sort2 = sort(v2);
sort3 = sort(v3);
t1 = length(sort2);
t2 = 1;
while sort2(t1)>sort3(t2) % compare the last to the first
    t1 = t1 - 1;
    t2 = t2 + 1;
end

threshold = (sort2(t1) + sort3(t2))/2;
xline(threshold)
miss_classify = length(find(v2 > threshold));
miss_classify = miss_classify + length(find(v3 < threshold));

% set_index1 = find(testlb == 2);
% set_index2 = find(testlb == 3);
% set_index = [set_index1 set_index2];
% s2 = zeros(1,length(set_index1));
% s3 = ones(1,length(set_index2));
% right = [s2 s3];
% testset = testdata(:,set_index);
% testnum = size(testset,2);
% U_1 = U(:,1:feature); % Add this in
% testmat = U_1'*testset;
% pval = w'*testmat;
% resvac = (pval>threshold); % 0 == digit 2, 1 == digit 3
% err = abs(resvac - right);
% rate = 1- (sum(err)/testnum)
%% LDA of 3 digits
% random sample 2000 out of the 60000
ind = randsample(60000,2000);
l = traininglb(ind); % label
feature = 200; % features we used
digits = S*V';
ind_1 = find(l == 1);
ind_2 = find(l == 2);
ind_3 = find(l == 3);
digit_1 = digits(1:feature,ind_1);
digit_2 = digits(1:feature,ind_2);
digit_3 = digits(1:feature,ind_3);
m1 = mean(digit_1,2);
m2 = mean(digit_2,2);
m3 = mean(digit_3,2);
% within-class variance
Sw = 0; 
for k = 1:length(ind_1)
    Sw = Sw + (digit_1(:,k) - m1)*(digit_1(:,k) - m1)';
end
for k = 1:length(ind_2)
    Sw = Sw + (digit_2(:,k) - m2)*(digit_2(:,k) - m2)';
end
for k = 1:length(ind_3)
    Sw = Sw + (digit_3(:,k) - m3)*(digit_3(:,k) - m3)';
end
% between class
Sb = 0;
mu = (m1+m2+m3)/3;
Sb = (m1 - mu)*(m1 - mu)';
Sb = Sb + (m2 - mu)*(m2 - mu)';
Sb = Sb + (m3 - mu)*(m3 - mu)';

[V2, D] = eig(Sb,Sw);
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);

v1 = w'*digit_1;
v2 = w'*digit_2;
v3 = w'*digit_3;

if mean(v2)>mean(v3)
    w = -w;
    v2 = -v2;
    v3 = -v3;
end

plot(v1, ones(length(ind_1)), 'mx', 'Linewidth',2), hold on
plot(v2,ones(length(ind_2))*0.5,'ob','Linewidth',2),
plot(v3,zeros(length(ind_3)),'dr','Linewidth',2)
% classifier between digit 1 and 2
sort1 = sort(v1);
sort2 = sort(v2);
t1 = length(sort1);
t2 = 1;
while sort1(t1)>sort2(t2) % compare the last to the first
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold = (sort1(t1) + sort2(t2))/2;
xline(threshold,'-m')
% % classifier between digit 1 and 2
% sort1 = sort(v1);
% sort3 = sort(v3);
% t1 = length(sort1);
% t2 = 1;
% while sort1(t1)>sort3(t2) % compare the last to the first
%     t1 = t1 - 1;
%     t2 = t2 + 1;
% end
% threshold = (sort1(t1) + sort3(t2))/2;
% xline(threshold,'-r')
% classifier between digit 2 and 3
sort2 = sort(v2);
sort3 = sort(v3);
t1 = length(sort2);
t2 = 1;
while sort2(t1)>sort3(t2) % compare the last to the first
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold = (sort2(t1) + sort3(t2))/2;
xline(threshold)
%% Reshape the testing data
test_images = double(reshape(test_images, size(test_images,1)*size(test_images,2), []).');
test_labels = double(test_labels);
% feature/mode = row
% image on column, 1 column is the image of digit
testdata = transpose(test_images);
testlb = transpose(test_labels);
%% train the data to test on the test data
feature = 200; % how many features we want to use
training_rate = zeros(10,10);
correct_rate_LDA = zeros(10,10);
miss_classify = 0;
for j = 1:10
    for k = 1:10
        miss_classify = 0;
        if j ~= k
            % build a classifier between digit m and digit n
            m = j - 1;
            n = k - 1;
            digits = S*V';
            nj = find(testlb == m);
            nk = find(testlb == n);
            digit_j = digits(1:feature,nj);
            digit_k = digits(1:feature,nk);
            mj = mean(digit_j,2);
            mk = mean(digit_k,2);
            
            % within-class variance
            Sw = 0; 
            for c = 1:length(nj)
                Sw = Sw + (digit_j(:,c) - mj)*(digit_j(:,c) - mj)';
            end
            for c = 1:length(nk)
                Sw = Sw + (digit_k(:,c) - mk)*(digit_k(:,c) - mk)';
            end
            % between class
            Sb = (mj-mk)*(mj-mk)'; 

            [V2, D] = eig(Sb,Sw);
            [lambda, ind] = max(abs(diag(D)));
            w = V2(:,ind);
            w = w/norm(w,2);

            vj = w'*digit_j;
            vk = w'*digit_k;

            if mean(vj)>mean(vk)
                w = -w;
                vj = -vj;
                vk = -vk;
            end
            
            sortj = sort(vj);
            sortk = sort(vk);
            t1 = length(sortj);
            t2 = 1;
            while sortj(t1)>sortk(t2) % compare the last to the first
                t1 = t1 - 1;
                t2 = t2 + 1;
            end
            threshold = (sortj(t1) + sortk(t2))/2;
            miss_classify = length(find(vj > threshold));
            miss_classify = miss_classify + length(find(vk < threshold));
            training_rate(j,k) = 1 - miss_classify/(length(vj) + length(vk));
            
            % Validate using test data
            set_index1 = find(testlb == m);
            set_index2 = find(testlb == n);
            set_index = [set_index1 set_index2];
            right = [zeros(1,length(set_index1)) ones(1,length(set_index2))];
            testset = testdata(:,set_index);
            testnum = size(testset,2);
            U_1 = U(:,1:feature); % Add this in
            testmat = U_1'*testset;
            pval = w'*testmat;
            resvac = (pval>threshold); % 0 == digit j, 1 == digit k
            err = abs(resvac - right);
            correct_rate_LDA(j,k) = 1 - (sum(err)/testnum);
        end
    end
end
%%
load("training_rate_LDA.mat")
load("correct_LDA.mat")
load("svm_result.mat")
%%

svm_result = zeros(11,11);
svm_result(:,1) = 0:10;
svm_result(1,:) = 0:10;
svm_result(2:11,2:11) = result;
%% Tree
tree=fitctree(trainingdata',traininglb','MaxNumSplits',10);
%%
1 - loss(tree,trainingdata',traininglb') % accuracy rate on trainingdata
1 - loss(tree,testdata',testlb') % accuracy rate on test data

%% SVM for digits 1 and 2
ind_1 = find(traininglb==0);
ind_2 = find(traininglb==1);
ind = [ind_1,ind_2];
X = trainingdata(:,ind);
y = [repelem(0,length(ind_1)) repelem(1,length(ind_2))];
Mdl = fitcsvm(X',y');
%%
ind_1 = find(testlb == 2);
ind_2 = find(testlb == 6);
ind = [ind_1,ind_2];
right = [repelem(0,length(ind_1)) repelem(1,length(ind_2))];
test_labels = predict(Mdl,X');
%%
testnum = length(y);
% diff = abs(right' - test_labels);
% testnum - sum(right'==test_labels);
correct = sum(y'==test_labels)/testnum
%% SVM all digits
result = zeros(10,10);
for j = 1:10
    for k = 1:10
        if (j ~= k)
            m = j-1;
            n = k-1;
            ind_m = find(traininglb==m);
            ind_n = find(traininglb==n);
            ind = [ind_m ind_n];
            X = trainingdata(:,ind);
            y = [repelem(m,length(ind_m)) repelem(n,length(ind_n))];
            Mdl = fitcsvm(X',y' );
            % validate the classifier using test data
            test_ind_m = find(testlb==m);
            test_ind_n = find(testlb==n);
            test_ind = [test_ind_m test_ind_n];
            right = [repelem(m,length(test_ind_m)) repelem(n,length(test_ind_n))];
            test_labels = predict(Mdl,testdata(:,test_ind)');
            testnum = length(test_ind);
            % record the correct rate
            result(j,k) = 1-(testnum-sum(right'==test_labels))/testnum;
        end 
    end
end
%% SVM results
load("svm_result.mat")
%%
function [images, labels] = mnist_parse(path_to_digits, path_to_labels)

% The function is curtesy of stackoverflow user rayryeng from Sept. 20,
% 2016. Link: https://stackoverflow.com/questions/39580926/how-do-i-load-in-the-mnist-digits-and-label-data-in-matlab

% Open files
fid1 = fopen(path_to_digits, 'r');

% The labels file
fid2 = fopen(path_to_labels, 'r');

% Read in magic numbers for both files
A = fread(fid1, 1, 'uint32');
magicNumber1 = swapbytes(uint32(A)); % Should be 2051
fprintf('Magic Number - Images: %d\n', magicNumber1);

A = fread(fid2, 1, 'uint32');
magicNumber2 = swapbytes(uint32(A)); % Should be 2049
fprintf('Magic Number - Labels: %d\n', magicNumber2);

% Read in total number of images
% Ensure that this number matches with the labels file
A = fread(fid1, 1, 'uint32');
totalImages = swapbytes(uint32(A));
A = fread(fid2, 1, 'uint32');
if totalImages ~= swapbytes(uint32(A))
    error('Total number of images read from images and labels files are not the same');
end
fprintf('Total number of images: %d\n', totalImages);

% Read in number of rows
A = fread(fid1, 1, 'uint32');
numRows = swapbytes(uint32(A));

% Read in number of columns
A = fread(fid1, 1, 'uint32');
numCols = swapbytes(uint32(A));

fprintf('Dimensions of each digit: %d x %d\n', numRows, numCols);

% For each image, store into an individual slice
images = zeros(numRows, numCols, totalImages, 'uint8');
for k = 1 : totalImages
    % Read in numRows*numCols pixels at a time
    A = fread(fid1, numRows*numCols, 'uint8');

    % Reshape so that it becomes a matrix
    % We are actually reading this in column major format
    % so we need to transpose this at the end
    images(:,:,k) = reshape(uint8(A), numCols, numRows).';
end

% Read in the labels
labels = fread(fid2, totalImages, 'uint8');

% Close the files
fclose(fid1);
fclose(fid2);
end

% https://stefansavev.com/blog/svd-the-projections-view/
% https://www.cs.utah.edu/~jeffp/teaching/cs5955/L15-SVD.pdf
% https://www.arcjournals.org/pdfs/ijsimr/v5-i4/4.pdf
