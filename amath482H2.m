
%% Floyd.m4a 
clear all; close all; clc

[y, Fs] = audioread('Floyd.m4a');
tr_fld = length(y)/Fs; % record time in seconds

% plot((1:length(y))/Fs,y);
% xlabel('Time [sec]'); ylabel('Amplitude');
% title("Comfortably Numb");
% p8 = audioplayer(y,Fs); playblocking(p8);
%%
y = y(3*(length(y)/4):4*(length(y)/4)); % change the portion to look at 
tr_fld = length(y)/Fs;

%% SPECTROGRAM 
t = transpose((1:length(y))/Fs);
a = 200;
tau = 0:0.1:tr_fld;
for j = 1:length(tau)
    g = exp(-a*(t - tau(j)).^2); % Gausian filter
    fl_filtered = g.*y; 
    fl_t = fft(fl_filtered);
    
    fl_t_sort = find(abs(fl_t)>0.01);
    fl_t = fl_t(fl_t_sort);
    fl_t = fl_t(1:100000);

    Sgt_spec(:,j) = fftshift(abs(fl_t)); % We don't want to scale it
end
 %% BASS 1
L = tr_fld;
n = length(y);
k = (1/L)*[0:n/2-1 -n/2:-1]; % frequency domain 
bass_filtered = lowpass(y,200,Fs);

% y_t = fft(y); % got the frequency information
% % fc = 250; % low pass filter with 250 Hz
% y_t(abs(k) > 200) = 0; % excludes those frequency are higher than 250 Hz
% y_filtered = ifft(y_t);
% plot((1:length(y))/Fs,y_filtered);
% % subplot(1,2,2)
% % title("Comfortably Numb's amplitude after filtering out high frequency");
% % xlabel('Time [sec]'); ylabel('Amplitude');
% % subplot(1,2,1)
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title("Comfortably Numb Original");

%[b,a] = butter(6, fc/(Fs/2));
%y_filtered = filter(b,a,y);
%% BASS 2
a = 200;
t = transpose((1:length(y))/Fs);
tau = 0:0.1:tr_fld;
for j = 1:length(tau)
    g = exp(-a*(t - tau(j)).^2);
    fl_filtered = g.*y_filtered;
    fl_t = fft(fl_filtered);
   
%     fl_t_sort = find(abs(fl_t)>0.0001); % want to excludes those value that are zero
%     fl_t = fl_t(fl_t_sort);
    fl_t = fl_t(1:100000); % add filter to filter out frequency that out of tone.
    
    bass_freq(:,j) = fftshift(abs(fl_t)); % We don't want to scale it
end
%% GUITAR 1 A
L = tr_fld;
n = length(y);
k = (1/L)*[0:n/2-1 -n/2:-1]; % frequency domain 

y_t = fft(y); % got the frequency information
% fc = 250; % low pass filter with 250 Hz
y_t(abs(k) < 200) = 0; % excludes those frequency are higher than 250 Hz
guitar_filtered = ifft(y_t);
%% GUITAR 1 B 
L = tr_fld;
n = length(y);
k = (1/L)*[0:n/2-1 -n/2:-1]; % frequency domain 

guitar_filtered = highpass(y, 150, Fs);
%% GUITAR 2
a = 200;
t = transpose((1:length(y))/Fs);
tau = 0:0.1:tr_fld;
for j = 1:length(tau)
    g = exp(-a*(t - tau(j)).^2);
    fl_filtered = g.*guitar_filtered;
    fl_t = fft(fl_filtered);
   
    fl_t_sort = find(abs(fl_t)>0.01); % want to excludes those value that are zero
    fl_t = fl_t(fl_t_sort);
    fl_t = fl_t(1:100000); % add filter to filter out frequency that out of tone.
    
    guitar_freq(:,j) = fftshift(abs(fl_t)); % We don't want to scale it
end

%% PLOT
L = tr_fld;
n = length(guitar_freq)+1;
k = (1/L)*[0:n/2-1 -n/2:-1]; % frequency domain 
ks = fftshift(k);
ks = ks(1:length(guitar_freq));
tau = 3*tr_fld:0.1:4*tr_fld;

pcolor(tau,ks,guitar_freq);
axis([3*tr_fld 4*tr_fld 150 1000]); % change the xlim to fit the portion of the data
shading interp
colormap(hot)
% 
% plot(tau, bass_freq)
title("Comfortably Numb");
xlabel('time (s)'), ylabel('frequency (Hz)')
% guitar a
% yline(400,'w');
% yline(558,'w');
% yline(305,'w');
% yline(258,'w');
% yline(204,'w');
% yline(481,'w');
% yline(801,'w');
% yline(247,'w');
% guitar b
yline(460,'w');
yline(258,'w');
yline(210,'w');
yline(365,'w');
yline(322,'w');
yline(628,'w');
yline(172,'w');
yline(550,'w');
yline(877,'w');

% bass
% yline(126,'w');
% yline(111,'w');
% yline(95,'w');
% yline(81,'w');
