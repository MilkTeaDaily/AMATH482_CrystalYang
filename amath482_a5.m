clear all; close all; clc
% Read in the video
v1 = VideoReader("ski_drop_low.mp4");
%% "ski_drop_low.mp4" video
j = 1;
while hasFrame(v1)
    I = readFrame(v1);
    I = rgb2gray(I);
    X(:,j) = reshape(I,[],1);
    j = j + 1;
end
X = im2double(X);
dt = v1.duration/(j-1);
t = linspace(0,v1.duration,size(X,2));
%%
X1 = X(:,1:end-1);
X2 = X(:,2:end);


[U, Sigma, V] = svd(X1,'econ');
rank = 1:300;
U = U(:,rank);
Sigma = Sigma(rank,rank);
V = V(:,rank);

S = U'*X2*V*diag(1./diag(Sigma)); % S tilda
[eV, D] = eig(S); % compute eigenvalues + eigenvectors
mu = diag(D); % extract eigenvalues
omega = log(mu)/dt;
Phi = U*eV; % eigenvector
%%
omega_abs = abs(omega);
zero_ind = find(omega_abs < 0.5);
Phi_bg = Phi(:,zero_ind);

%%
subplot(1,2,1)
plot(diag(Sigma),'ko')
title("Singular value")
ylabel('Eigenvalue')

subplot(1,2,2)
% make axis lines
line = -15:15;

plot(zeros(length(line),1),line,'k','Linewidth',2) % imaginary axis
hold on
plot(line,zeros(length(line),1),'k','Linewidth',2) % real axis
plot(real(omega)*dt,imag(omega)*dt,'r.','Markersize',15)
xlabel('Re(\omega)')
ylabel('Im(\omega)')
%% background
y0 = Phi_bg\X1(:,1);

u_modes = zeros(length(y0),length(t));
for iter = 1:length(t)
   u_modes(:,iter) = y0.*exp(omega(zero_ind)*t(iter)); 
end
u_background = Phi_bg*u_modes;

%% foreground
u_foreground = X - abs(u_background) + 0.5; % to make contrast, add some number to raise the tone of 'background'

%% looking at the background
im = reshape(u_foreground(:,61),[540,960]);
imshow(im)
%% looking at the foreground
im = reshape(X(:,300),[540,960]); % t = 0s
imshow(im); drawnow;
%%
% https://arxiv.org/pdf/1409.6358.pdf




%% "monte_carlo_low.mp4" video
clear all; close all;clc
v2 = VideoReader("monte_carlo_low.mp4");
j = 1;
while hasFrame(v2)
    I = readFrame(v2);
    I = rgb2gray(I);
    X(:,j) = reshape(I,[],1);
    j = j + 1;
end
X = im2double(X);
dt = v2.duration/(j-1);
t = linspace(0,v2.duration,size(X,2));
%%
X1 = X(:,1:end-1);
X2 = X(:,2:end);

[U, Sigma, V] = svd(X1,'econ');
rank = 1:300;
U = U(:,rank);
Sigma = Sigma(rank,rank);
V = V(:,rank);

S = U'*X2*V*diag(1./diag(Sigma)); % S tilda
[eV, D] = eig(S); % compute eigenvalues + eigenvectors
mu = diag(D); % extract eigenvalues
omega = log(mu)/dt;
Phi = U*eV; % eigenvector

%%
omega_abs = abs(omega);
zero_ind = find(omega_abs < 0.5);
Phi_bg = Phi(:,zero_ind);
%%
y0 = Phi_bg\X1(:,1); % pseudoinverse to get initial conditions

u_modes = zeros(length(y0),length(t));
for iter = 1:length(t)
   u_modes(:,iter) = y0.*exp(omega(zero_ind)*t(iter)); 
end
u_background = Phi_bg*u_modes;

%% foreground
u_foreground = (X - abs(u_background));

%% looking at the background
im = reshape(X(:,320),[540,960]);
imshow(im)
%% looking at the foreground
im = reshape(u_background(:,320),[540,960]); % t = 0s
imshow(im); drawnow;
%%
for j = 1:length(t)
    im = reshape(u_foreground(:,j),[540,960]);
    imshow(im); drawnow;
end
