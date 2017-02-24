clear; clc

im = imread('img/flower1.jpg');

bins = 8;
angle = 180;
level = 3;
roi = [1; 225; 1; 300];
p = computePHOG(im, bins, angle, level, roi)
