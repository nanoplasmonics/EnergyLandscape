% Load the data as a vector using readmatrix
data = readmatrix(''); %path to data
data = data(:, 1);
l = length(data);

y = (data + 0.002441 * rand(l, 1))- mean(data) + 0.5;

% Compute histogram with 100 bins
[counts, edges] = histcounts(y, 100);
bin_centers = (edges(1:end-1) + edges(2:end)) / 2;



figure;
plot(bin_centers, counts)
xlabel('Transmission (V)')

% Padding to have 128 (2^7)
pp = zeros(1, 128);
pp(15:114) = counts; 

% Gaussian 
sd = 5; 
x = 1:128;
g = 1 / (2 * pi * sd) * exp(-(x - 64.5).^2 / (2 * sd^2));

% Deconvolution
V = 0.0001;
luc_5 = deconvlucy(pp, g, 51, sqrt(V));

% Energy landscape
U_5 = -1 * log(luc_5);

% Plot deconvolution and energy landscape together
figure;
yyaxis left;
plot(bin_centers, luc_5(15:114)/max(luc_5(15:114)), '-r', 'LineWidth', 3);
xlabel('Voltage');
ylabel('Probability Density');

yyaxis right;
plot(bin_centers, U_5(15:114), '-k','LineWidth', 3);
ylabel('Free Energy (k_{B}T');



% Gaussian 
sd = 5.5; 
x = 1:128;
g = 1 / (2 * pi * sd) * exp(-(x - 64.5).^2 / (2 * sd^2));

% Deconvolution
V = 0.0001;
luc_6 = deconvlucy(pp, g, 51, sqrt(V));

% Energy landscape
U_6 = -1 * log(luc_6);

% Plot deconvolution and energy landscape together
figure;
yyaxis left;
plot(bin_centers, luc_6(15:114)/max(luc_6(15:114)), '-r', 'LineWidth', 3);
xlabel('Voltage');
ylabel('Probability Density');

yyaxis right;
plot(bin_centers, U_6(15:114), '-k','LineWidth', 3);
ylabel('Free Energy (k_{B}T');



% Gaussian 
sd = 6; 
x = 1:128;
g = 1 / (2 * pi * sd) * exp(-(x - 64.5).^2 / (2 * sd^2));

% Deconvolution
V = 0.0001;
luc_7 = deconvlucy(pp, g, 51, sqrt(V));

% Energy landscape
U_7 = -1 * log(luc_7);

% Plot deconvolution and energy landscape together
figure;
yyaxis left;
plot(bin_centers, luc_7(15:114)/max(luc_7(15:114)), '-r', 'LineWidth', 3);
xlabel('Voltage');
ylabel('Probability Density');

yyaxis right;
plot(bin_centers, U_7(15:114), '-k','LineWidth', 3);
ylabel('Free Energy (k_{B}T');


% Prepare data for saving
data_to_save = [bin_centers', (luc_5(15:114)/max(luc_5(15:114)))', U_5(15:114)', (luc_6(15:114)/max(luc_6(15:114)))', U_6(15:114)', (luc_7(15:114)/max(luc_7(15:114)))', U_7(15:114)'];

% Write data to a text file
writematrix(data_to_save, 'E:\Temperature Files\dimer_power_4.txt', 'Delimiter', 'tab');
