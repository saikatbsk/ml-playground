%% ========================================================================
%% The Kmean algorithm. Tested with Octave.
%%
%% Inputs:
%%      K       - Number of clusters.
%%      X       - Training set, {x(1); x(2); x(3); ...; x(m)}; where each
%%                x(i) is a point in space. X is a m*2 matrix.
%%      MAXITR  - Maximum number of iterations.
%%
%% Output:
%%      idx     - Cluster indices, This is a numeric column vector.
%%                idx has as many rows as X, and each row indicates the
%%                cluster assignment of the corresponding observation.
%%      centers - Column vector containing cluster centers. K*2 matrix.
%% ========================================================================

function [idx centers] = Kmean(K, X, MAXITR)
    m = size(X, 1);     % Number of sample points

    if(K > m)           % Breaking condition
        fprintf('>>> Error! K must be less than the number of sample points.\n\n'); fflush(stdout);
    else
        % Randomly initialize K cluster centers
        centers = zeros(K, 2);
        perm = randperm(m);
        perm = perm(1:K);
        centers = double(X(perm, :));

        % initializing some more variables
        d = zeros(K, 1);
        idx = zeros(m, 1);

        % K-means clustering
        fprintf('Begin K-mean clustering..'); fflush(stdout);

        for n = 1:MAXITR
            % Step 1 - Cluster assignments
            for i = 1:m
                x = [X(i, :)];

                for k = 1:K
                    c = [centers(k,:)];
                    d(k) = ((x(1) - c(1))^2) + ((x(2) - c(2))^2);
                end

                [minval idx(i)] = min(d);
            end

            % Step 2 - Move cluster centers
            for k = 1:K
                cluster_points = [];

                for i = 1:m
                    if(idx(i) == k)
                        cluster_points = [cluster_points; X(i, :)];
                    end
                end

                if (size(cluster_points, 1) > 1)
                    centers(k, :) = mean(cluster_points);
                else
                    centers(k, :) = cluster_points(:);
                end
            end
        end

        fprintf('Done.\n'); fflush(stdout);
    end
end
