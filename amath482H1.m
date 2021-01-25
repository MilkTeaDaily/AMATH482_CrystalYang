clear all; close all; clc
load subdata.mat

L = 10;
n = 64;
x2 = linspace(-L,L,n+1); x = x2(1:n); y =x; z = x;

k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k); % Create the frequency domain

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

for j=1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    M = max(abs(Un),[],'all');
%     close all, isosurface(X,Y,Z,abs(Un)/M,0.7)
%     axis([-20 20 -20 20 -20 20]), grid on, drawnow
    pause(1)
end

%% Averaging the signal
ave(:,:,:) = zeros(n,n,n);

for j = 1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    Unt = fftn(Un); % fft1() when only one dinmension
    ave = ave + Unt;
end
ave = abs(fftshift(ave))/49;
m = max(abs(ave),[],'all');
isosurface(Kx,Ky,Kz,abs(ave)/m,0.7);
xlabel('Kx')
ylabel('Ky')
zlabel('Kz')
axis([-20 20 -20 20 -20 20]), grid on, drawnow

[m,ind] = max(ave(:));
[ind_x,ind_y,ind_z] = ind2sub([n,n,n],ind);
center_Kx = Kx(ind_x,ind_y,ind_z); 
center_Ky=  Ky(ind_x,ind_y,ind_z); 
center_Kz = Kz(ind_x,ind_y,ind_z);


%% Filtering the signal and getting the coordinate 
tau = 0.2;
filter = fftshift(exp(-tau*((Kx-center_Kx).^2 + (Ky-center_Ky).^2 + (Kz-center_Kz).^2)));

for j=1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
   
    Untf = fftshift(filter.*fftn(Un)); % apply a filter in the time spectrum 
    Unf = ifftn(Untf);
    M = max(abs(Unf),[],'all'); % for normalization
    
    % Capture the location of the submarine
    [m,ind] = max(Unf(:));
    [ind_x,ind_y,ind_z] = ind2sub([n,n,n],ind);
    Point_x(j) = Kx(ind_x,ind_y,ind_z); 
    Point_y(j) =  Ky(ind_x,ind_y,ind_z); 
    
    isosurface(X,Y,Z,abs(Unf)/M,0.9), hold on
    axis([-20 20 -20 20 -20 20]), grid on, drawnow
    pause(1)
end

%%
isosurface(Kx,Ky,Kz,filter,0.6)
axis([-20 20 -20 20 -20 20]), grid on, drawnow



