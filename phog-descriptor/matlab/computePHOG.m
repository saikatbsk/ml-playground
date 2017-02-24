% --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- %
% Fucntion to compute Pyramid Histogram of Oriented Gradients
%
% About PHOG:
%       The PHOG descriptor consists of a histogram of orientation gradients over
%       each image subregion at each resolution level - a Pyramid of Histograms
%       of Orientation Gradients (PHOG). The distance between two PHOG image
%       descriptors then reflects the extent to which the images contain similar
%       shapes and correspond in their spatial layout.
%       ref: http://www.robots.ox.ac.uk/~vgg/research/caltech/phog.html
%
% INPUTS:
%       im     - Input image (rgb or gray)
%       bins   - Number of bins on the histogram
%       angle  - Angle, either 180 or 360
%       levels - Number of pyramid levels
%       roi    - Region of interest (ytop, ybottom, xleft, xright)
%
% OUTPUT:
%       p      - Pyramid histogram of oriented gradients
% --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- %

function p = computePHOG(im, bins, angle, levels, roi)
    % Convert rgb input image to gray
    if size(im, 3) == 3
        im = rgb2gray(im);
    end

    mh = [];
    mg = [];

    % If sum of all the pixel values is grater than certain value, say 100, then
    % it means the image is not entirely black and we can continue processing it
    if sum(sum(im)) > 100
        E = edge(im, 'canny');

        [GX, GY] = gradient(double(im));
        Gr = sqrt((GX.*GX) + (GY.*GY));

        index = GX == 0;
        GX(index) = 1e-5;

        YX = GY./GX;
        if angle == 180
            A = ((atan(YX)+(pi/2)) * 180)/pi;
        end
        if angle == 360
            A = ((atan2(GY, GX)+pi) * 180)/pi;
        end

        [mh, mg] = binMatrix(A, E, Gr, angle, bins);
    else
        mh = zeros(size(im, 1), size(im, 2));
        mg = zeros(size(im, 1), size(im, 2));
    end

    mh_roi = mh(roi(1, 1):roi(2, 1), roi(3, 1):roi(4, 1));
    mg_roi = mg(roi(1, 1):roi(2, 1), roi(3, 1):roi(4, 1));

    p = phogDescriptor(mh_roi, mg_roi, levels, bins);
end
