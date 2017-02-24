% Simple neural network in Octave

function main
    X = [ 0 0 1 ; 1 1 1 ; 1 0 1 ; 0 1 1 ];
    Y = [ 0 ; 1 ; 1 ; 0 ];

    randn('seed', 1);

    syn0 = 2 * randn(3, 1) - 1;

    for i = 1:10000
        l0 = X;
        l1 = sigmoid(l0 * syn0);

        l1_del = diag(Y - l1) * sigmoid(l1, 1);

        syn0 = syn0 + (l0' * l1_del);
    end

    disp('Output after training:');
    disp(l1);
end

function sg = sigmoid(t, deriv)
    if nargin < 2 || deriv == 0
        sg = 1 ./ (1 + exp(-t));
    elseif deriv == 1
        sg = diag(t) * (1 - t);
    end
end
