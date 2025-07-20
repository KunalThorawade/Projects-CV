s = load('clean_speech.txt');
s_v = load('noisy_speech.txt');
w = load('external_noise.txt');

% filter parameters
fs = 44100;                      % sampling frequency
N = 12;                          % N tap filter
lambda = 0.99999;                     % forgetting factor
delta = 1e5;                         % constant for initialising the S_D matrix as diagonal matrix

% notch filter parameters
f_n = [1000];                        % tonal frequency components in the noise to be retained 
mode = false;                        % select the partial supression or full supression mode
M = 3 ;                              % notch filter order 
f_partial = repmat(f_n, 1, M);       % repeat 2nd order filter for same frequency M times
n = length(s_v);                     % number of samples

if mode
    % notch filter coefficients initialization and calculation  
    p_n = length(f_partial);
    coeff_b = cell(p_n, 1);
    coeff_a = cell(p_n, 1);
    w_buffer = zeros(p_n, 2);      % [w(n-1), w(n-2)] input signal samples
    x_buffer = zeros(p_n, 2);      % [x(n-1), x(n-2)] input signal componenets with tonal freq removed
    
    for i = 1:p_n
        omega = 2 * pi * f_partial(i) / fs;
        r = 0.999;
        coeff_b{i} = [1, -2*cos(omega), 1];     % numerator coefficients
        coeff_a{i} = [1, -2*r*cos(omega), r^2]; % denominator coefficients
    end
end

% initialising the elements required for RLS
buffer = zeros(N, 1);           % buffer for input values for RLS algorithm
S_D = delta * eye(N);               % S_D(-1) initialization for recursive implementation 
p_D = zeros(N,1);               % p_D(-1) initialization
h = zeros(N,1);                 % adaptive filter coefficients 

v_hat = zeros(n,1);             % estimated noise 
e = zeros(n,1);                 % error signal (output signal)

% adaptive filtering using RLS algorithm
for k = 1:n
    
    if mode
        % apply each filter in sequence
        for i = 1:p_n
            % get the parameters and required values for the notch filter for the frequency
            b = coeff_b{i};
            a = coeff_a{i};
            w_prev = w_buffer(i,:)';
            x_prev = x_buffer(i,:)';
           
            % apply the filter to the current sample
            if k == 1
                x = b(1) * w(k);
            elseif k == 2
                x = b(1)*w(k) + b(2)*w_prev(1) - a(2)*x_prev(1);
            else
                x = b(1)*w(k) + b(2)*w_prev(1) + b(3)*w_prev(2) - a(2)*x_prev(1) - a(3)*x_prev(2);
            end
            
            % update the states for next iteration
            w_buffer(i,:) = [w(k), w_prev(1)];
            x_buffer(i,:) = [x, x_prev(1)];            
            
            w(k) = x; %the tonal frequency is removed from input signal passed to the adaptive filter
        end
    end
    
    % RLS 
    buffer(2:end) = buffer(1:end-1);  % Shift buffer for latest N samples
    buffer(1) = w(k);                 % Add filtered reference sample
    xk = buffer(1:N);                 % input vector to the adaptive filter
    
    denom = lambda + xk.' * S_D * xk;
    S_D = (1/lambda) * (S_D - (S_D * xk * (xk.') * S_D) / denom);
    p_D = lambda * p_D + s_v(k) * xk;
    h = S_D * p_D;
    v_hat(k) = h.' * xk;
    e(k) = s_v(k) - v_hat(k);
end

% SNR calculation
snr_1 = 10 * log10(mean(s.^2) / mean((s_v - s).^2));
snr_2 = 10 * log10(mean(s.^2) / mean((e - s).^2));
fprintf('SNR for the given noisy speech: %.4f dB \n ',snr_1);
fprintf('SNR after the filtering operation: %.4f dB\n ',snr_2);


% FFT for comparison 
fft_clean = abs(fft(s));
fft_noisy = abs(fft(s_v));
fft_error = abs(fft(e));
fft_res_noise = abs(fft(e-s));

% frequency vector
freq = fs * (0:(length(s)-1)) / length(s);

figure('Position', [100, 100, 1000, 800]);  


subplot(2, 2, 1);
plot(freq, 20*log10(fft_clean));
title('Clean Speech');
xlabel('Frequency [Hz]');
ylabel('Magnitude');
xlim([0 fs/2]);
grid on;

subplot(2, 2, 2);
plot(freq, 20*log10(fft_noisy));
title('Noisy Speech');
xlabel('Frequency [Hz]');
ylabel('Magnitude');
xlim([0 fs/2]);
grid on;

subplot(2, 2, 3);
plot(freq, 20*log10(fft_error));
title('Estimated Clean Speech');
xlabel('Frequency [Hz]');
ylabel('Magnitude');
xlim([0 fs/2]);
grid on;

subplot(2, 2, 4);
plot(freq, 20*log10(fft_res_noise));
title('Residual Noisy Speech');
xlabel('Frequency [Hz]');
ylabel('Magnitude');
xlim([0 fs/2]);
grid on;

sgtitle('FFTs Analysis of Speech Signals');