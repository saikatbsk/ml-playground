% --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- %
% Computes a Matrix (mh) with the same size of the image where (i,j) position
% contains the histogram value for the pixel at position (i,j); and another
% matrix (mg) where the position (i,j) contains the gradient value for the pixel
% at position (i,j)
%
% INPUTS:
%       A     - Angle values
%       E     - Edge image
%       Gr    - Matrix containing gradient values
%       angle - Angle, either 180 or 360
%       bins  - Number of bins on the histogram
%
% OUTPUT:
%       mh    - Matrix with the histogram values
%       mg    - Matrix with the gradient values (only for the pixels belonging
%               to an edge)
% --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- %

function [mh, mg] = binMatrix(A, E, Gr, angle, bins)
    % Find label matrix containing labels for 8-connected objects found in E
    [L, n] = bwlabel(E);

    rows = size(E, 1);
    cols = size(E, 2);

    mh = zeros(rows, cols);
    mg = zeros(rows, cols);

    nAngle = angle/bins;

    % For all 8-connected objects
    for i = 1:n
        % Find the position co-ordinates where the i-th object is located
        [posY, posX] = find(L == i);

        for j = 1:length(posY)
            % Get each individual (x, y) co-ordinate
            pos_y = posY(j);
            pos_x = posX(j);

            b = ceil(A(pos_y, pos_x) / nAngle);

            if Gr(pos_y, pos_x) > 0
                mh(pos_y, pos_x) = b;
                mg(pos_y, pos_x) = Gr(pos_y, pos_x);
            end
        end
    end
end
