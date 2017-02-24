% Testing Kmean algorithm. Tested with Octave.

clear; clc;

K = 3;
% X = 100.*rand(90, 2);

X = [randn(100,2)*0.75+ones(100,2);
    randn(100,2)*0.5-ones(100,2)];

[idx centers] = Kmean(K, X, 10);

figure;
hold on
title 'Kmean Clustering Example';
plot(X(idx==1, 1), X(idx==1, 2), 'r.', 'MarkerSize', 6);
plot(X(idx==2, 1), X(idx==2, 2), 'g.', 'MarkerSize', 6);
plot(X(idx==3, 1), X(idx==3, 2), 'b.', 'MarkerSize', 6);
plot(centers(:, 1), centers(:, 2), 'kx', 'MarkerSize', 12, 'LineWidth', 3);
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroids', 'Location', 'NorthWest');
hold off
