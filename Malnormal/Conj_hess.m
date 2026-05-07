% Script to calculate the malnormality constant of the matrix 
% A = T_n + 2D_n
% using the Hessian algorithm. Returns the smallest eigenvalue of H.
% Calls the 'give_mal' function.
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
    % display T_n

    % create array of rotation angles
    thetas = zeros(1,n);
    for k = 1:n
        thetas(1, k) = (pi*k)/n;
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
    display(A);

    % Calculate the malnormality constant using the Hessian algorithm. 
    % Needs to access the scripts in the 'Hessian' folder, so may need to
    % run addpath('Hessians') in the terminal
    % mal_T = give_mal(A, 2*n);
    % display(2*n);
    % display(mal_T);
end