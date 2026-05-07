% Calculate the malnormality constant of matrix
% A = T_n + 2D_n
% using MATLAB's optimization toolbox, specifically the interior-point
% algorithm of the fmincon function. Calls the 'give_mal_opt' function.
% Is designed to run multiple trials of the same matrix, if desired.

for n = 2:4

    % create T_n
    T = zeros(2*n, 2*n);
    for c = 1:2*n
        for r = 1:2*n
            if (c == r) && (c <= 2*n - 2)
                T(r, c+2) = 1;
            elseif (c == 1) && (r == 2*n - 1)
                T(r, c) = 1;
            elseif (c == 2) && (r == 2*n)
                T(r,c) = 1;
            end
        end
    end
    
    % create array of rotation angles
    thetas = zeros(1,n);
    for k = 1:n
        thetas(1, k) = (2*pi*k)/n;
    end
    
    % create D_n
    D = zeros(2*n, 2*n);
    tracker = 1;
    for c = 1:2:(2*n - 1)
        for r = 1:2:(2*n - 1)
            if c == r
                D(c, r) = cos(thetas(1,tracker));
                D(c+1, r) = sin(thetas(1,tracker));
                D(c, r+1) = -sin(thetas(1,tracker));
                D(c+1, r+1) = cos(thetas(1,tracker));
                tracker = tracker + 1;
            end
        end
    end
    % display(D);
    
    A = T + 2*D;
    % display(A);

    num_trials = 2;
    mal = zeros(num_trials, 1);
    for trial = 1:num_trials
        mal(trial, 1) = give_mal_opt(A, 2*n);
        display(mal(trial, 1));
        display(trial);
    end
end

figure
histogram(mal);


% figure
% histfit(mal,10,'Kernel');

% figure
% pdMal = fitdist(mal, 'Kernel');
% x = min(mal):(max(mal)/100):max(mal);
% yMal = pdf(pdMal, x);
% plot(x, yMal,'k-','LineWidth',2);